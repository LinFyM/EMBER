"""Policy-layer trace reader and axis-aligned PI05 LoRA hyperdecoder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class FewShotM2PError(RuntimeError):
    """Raised when the K4 trace or public PI05 topology changes."""


class RoutedZeroPreservingBlock(torch.nn.Module):
    """Use coordinate routes only in Q/K while all dynamic value stays video-owned."""

    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        if min(width, heads, expansion) <= 0 or width % heads:
            raise FewShotM2PError("invalid zero-preserving transformer block")
        self.heads = int(heads)
        self.head_width = width // heads
        self.norm_attention = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.norm_ffn = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width, bias=False),
        )

    def forward(self, value: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or route.shape != value.shape:
            raise FewShotM2PError("invalid routed axis block input")
        normalized = self.norm_attention(value)
        batch, tokens, width = value.shape
        q = self.query(normalized + route).reshape(
            batch, tokens, self.heads, self.head_width
        ).transpose(1, 2)
        k = self.key(normalized + route).reshape(
            batch, tokens, self.heads, self.head_width
        ).transpose(1, 2)
        v = self.value(value).reshape(
            batch, tokens, self.heads, self.head_width
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(batch, tokens, width)
        value = value + self.output(attended)
        return value + self.ffn(self.norm_ffn(value))


@dataclass(frozen=True)
class PolicyTargetSpec:
    """One PI05 public module with paired physical A/B tensors."""

    module_index: int
    module: str
    a_name: str
    b_name: str
    rank: int
    input_width: int
    output_width: int


@dataclass(frozen=True)
class PolicyLayerGroup:
    """One action-in, action-expert layer, or action-out output group."""

    name: str
    targets: tuple[PolicyTargetSpec, ...]


def build_policy_target_specs(
    tensor_specs: Sequence[object],
) -> tuple[PolicyTargetSpec, ...]:
    grouped: dict[int, dict[int, object]] = {}
    for item in tensor_specs:
        grouped.setdefault(int(getattr(item, "module_index")), {})[
            int(getattr(item, "factor_index"))
        ] = item
    if not grouped or any(set(pair) != {0, 1} for pair in grouped.values()):
        raise FewShotM2PError("policy target A/B pairing changed")
    result = []
    for module_index in sorted(grouped):
        a = grouped[module_index][0]
        b = grouped[module_index][1]
        if getattr(a, "module") != getattr(b, "module") or getattr(a, "rank") != getattr(b, "rank"):
            raise FewShotM2PError("policy target module pairing changed")
        result.append(
            PolicyTargetSpec(
                module_index=module_index,
                module=str(getattr(a, "module")),
                a_name=str(getattr(a, "name")),
                b_name=str(getattr(b, "name")),
                rank=int(getattr(a, "rank")),
                input_width=int(getattr(a, "width")),
                output_width=int(getattr(b, "width")),
            )
        )
    return tuple(result)


_EXPERT_TARGET = re.compile(
    r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
)


def build_policy_layer_groups(
    targets: Sequence[PolicyTargetSpec],
    *,
    expert_layers: int,
) -> tuple[PolicyLayerGroup, ...]:
    action_in: PolicyTargetSpec | None = None
    action_out: PolicyTargetSpec | None = None
    layers: dict[int, dict[str, PolicyTargetSpec]] = {}
    for target in targets:
        if target.module.endswith("action_in_proj"):
            action_in = target
            continue
        if target.module.endswith("action_out_proj"):
            action_out = target
            continue
        match = _EXPERT_TARGET.fullmatch(target.module)
        if match is None:
            raise FewShotM2PError(f"unsupported PI05 policy target: {target.module}")
        layers.setdefault(int(match.group(1)), {})[match.group(2)] = target
    if (
        action_in is None
        or action_out is None
        or set(layers) != set(range(expert_layers))
        or any(set(pair) != {"q_proj", "v_proj"} for pair in layers.values())
    ):
        raise FewShotM2PError("PI05 policy-layer grouping changed")
    return (
        PolicyLayerGroup("action_in", (action_in,)),
        *(
            PolicyLayerGroup(
                f"expert_layer_{layer:02d}",
                (layers[layer]["q_proj"], layers[layer]["v_proj"]),
            )
            for layer in range(expert_layers)
        ),
        PolicyLayerGroup("action_out", (action_out,)),
    )


class PolicyLayerTraceM2P(torch.nn.Module):
    """Read unordered K4 policy traces and directly emit the public PI05 LoRA."""

    def __init__(
        self,
        groups: tuple[PolicyLayerGroup, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        width: int,
        memory_slots: int,
        temporal_terms: int,
        heads: int,
        blocks: int,
        ffn_expansion: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            not groups
            or min(width, memory_slots, temporal_terms, heads, blocks, ffn_expansion) <= 0
            or width % heads
            or blocks != 4
        ):
            raise FewShotM2PError("invalid layer-trace M2P topology")
        self.groups = groups
        self.width = int(width)
        self.memory_slots = int(memory_slots)
        self.temporal_terms = int(temporal_terms)
        self.heads = int(heads)
        self.head_width = width // heads

        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def route(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.group_route = route(len(groups))
        self.slot_route = route(memory_slots)
        self.temporal_route = route(temporal_terms)
        self.query_norm = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.trace_norm = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.group_output_weight = torch.nn.Parameter(
            torch.zeros(len(groups), width, width)
        )
        self.axis_blocks = torch.nn.ModuleList(
            RoutedZeroPreservingBlock(width, heads, ffn_expansion)
            for _ in range(blocks)
        )

        self._template_names: dict[str, str] = {}
        self._group_tensor_names: tuple[tuple[str, ...], ...] = tuple(
            tuple(
                name
                for target in group.targets
                for name in (target.a_name, target.b_name)
            )
            for group in groups
        )
        public_names = {name for names in self._group_tensor_names for name in names}
        if public_names != set(template_state):
            raise FewShotM2PError("policy-layer groups do not cover public LoRA")
        for index, name in enumerate(sorted(public_names)):
            buffer_name = f"template_{index:03d}"
            self.register_buffer(
                buffer_name,
                template_state[name].detach().contiguous(),
                persistent=True,
            )
            self._template_names[name] = buffer_name
        capacity = memory_slots * width
        for names in self._group_tensor_names:
            required = sum(getattr(self, self._template_names[name]).numel() for name in names)
            if required > capacity:
                raise FewShotM2PError("policy-layer memory cannot hold target tensors")

    def _read_traces(
        self,
        video_traces: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if (
            video_traces.ndim != 4
            or video_traces.shape[1:]
            != (len(self.groups), self.temporal_terms, self.width)
            or condition_video_offsets.ndim != 1
            or condition_video_offsets.dtype != torch.long
            or condition_video_offsets.numel() < 2
        ):
            raise FewShotM2PError("invalid K4 layer-trace descriptors")
        offsets = tuple(int(value) for value in condition_video_offsets.detach().cpu().tolist())
        if (
            offsets[0] != 0
            or offsets[-1] != video_traces.shape[0]
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise FewShotM2PError("invalid condition-video offsets")
        shot_counts = {right - left for left, right in zip(offsets, offsets[1:])}
        if len(shot_counts) != 1 or next(iter(shot_counts)) <= 1:
            raise FewShotM2PError("layer-trace M2P requires one fixed K greater than one")
        shots = next(iter(shot_counts))
        conditions = len(offsets) - 1
        traces = video_traces.reshape(
            conditions,
            shots,
            len(self.groups),
            self.temporal_terms,
            self.width,
        ).permute(0, 2, 1, 3, 4).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.width,
        )
        group_route = self.group_route[:, None]
        query_route = group_route + self.slot_route[None]
        query_route = query_route[None].expand(conditions, -1, -1, -1).reshape(
            conditions * len(self.groups), self.memory_slots, self.width
        )
        key_route = group_route + self.temporal_route[None]
        key_route = key_route[:, None].expand(-1, shots, -1, -1).reshape(
            len(self.groups), shots * self.temporal_terms, self.width
        )
        key_route = key_route[None].expand(conditions, -1, -1, -1).reshape_as(traces)

        q = self.query(self.query_norm(query_route)).reshape(
            conditions * len(self.groups), self.memory_slots, self.heads, self.head_width
        ).transpose(1, 2)
        normalized_traces = self.trace_norm(traces)
        k = self.key(normalized_traces + key_route).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        v = self.value(traces).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(
            conditions,
            len(self.groups),
            self.memory_slots,
            self.width,
        )
        return torch.einsum(
            "cgsw,gow->cgso",
            attended,
            self.group_output_weight,
        )

    def _axis_m2p(self, memory: torch.Tensor) -> torch.Tensor:
        conditions, groups, slots, width = memory.shape
        coordinate_route = self.group_route[:, None] + self.slot_route[None]
        for index, block in enumerate(self.axis_blocks):
            if index % 2 == 0:
                value = memory.permute(0, 2, 1, 3).reshape(
                    conditions * slots, groups, width
                )
                route = coordinate_route.permute(1, 0, 2)[None].expand(
                    conditions, -1, -1, -1
                ).reshape_as(value)
                memory = block(value, route).reshape(
                    conditions, slots, groups, width
                ).permute(0, 2, 1, 3)
            else:
                value = memory.reshape(conditions * groups, slots, width)
                route = coordinate_route[None].expand(
                    conditions, -1, -1, -1
                ).reshape_as(value)
                memory = block(value, route).reshape(
                    conditions, groups, slots, width
                )
        return memory

    def encode(
        self,
        video_traces: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> torch.Tensor:
        memory = self._axis_m2p(
            self._read_traces(video_traces, condition_video_offsets)
        )
        expected = (
            condition_video_offsets.numel() - 1,
            len(self.groups),
            self.memory_slots,
            self.width,
        )
        if memory.shape != expected or not bool(torch.isfinite(memory).all()):
            raise FewShotM2PError("policy-layer memory changed shape")
        return memory

    def decode(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            memory.ndim != 4
            or memory.shape[1:] != (len(self.groups), self.memory_slots, self.width)
            or not bool(torch.isfinite(memory).all())
        ):
            raise FewShotM2PError("invalid policy-layer memory")
        result: dict[str, torch.Tensor] = {}
        for group_index, names in enumerate(self._group_tensor_names):
            flat = memory[:, group_index].flatten(1)
            cursor = 0
            for name in names:
                template = getattr(self, self._template_names[name])
                stop = cursor + template.numel()
                dynamic = flat[:, cursor:stop].reshape(memory.shape[0], *template.shape)
                result[name] = dynamic.to(template.dtype) + template[None]
                cursor = stop
        return result

    def forward(
        self,
        video_traces: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(video_traces, condition_video_offsets))


def count_parameters(module: torch.nn.Module) -> int:
    """Return trainable parameter count for the architecture contract."""

    return sum(value.numel() for value in module.parameters() if value.requires_grad)
