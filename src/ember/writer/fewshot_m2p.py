"""Evidence-factorized policy-layer reader and PI05 LoRA hyperdecoder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class FewShotM2PError(RuntimeError):
    """Raised when the K4 trace or public PI05 topology changes."""


_EVIDENCE_FLOOR = 1e-8


def factorize_trace_evidence(
    physical_traces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate trace direction from bounded energy and K-shot consensus evidence."""

    if physical_traces.ndim != 5 or physical_traces.shape[1] <= 1:
        raise FewShotM2PError("trace evidence requires [condition,K,group,term,width]")
    working = physical_traces.float()
    direction = F.normalize(working, dim=-1, eps=1e-12)
    token_energy = working.square().sum(dim=-1)
    group_energy = token_energy.sum(dim=-1)
    video_energy = group_energy.sum(dim=-1)
    group_share = torch.where(
        video_energy[..., None] > 0,
        group_energy / video_energy[..., None].clamp_min(_EVIDENCE_FLOOR),
        torch.zeros_like(group_energy),
    )
    frequency_share = torch.where(
        group_energy[..., None] > 0,
        token_energy / group_energy[..., None].clamp_min(_EVIDENCE_FLOOR),
        torch.zeros_like(token_energy),
    )

    log_scale = -torch.log(
        torch.tensor(
            _EVIDENCE_FLOOR,
            dtype=working.dtype,
            device=working.device,
        )
    )

    def bounded_log_share(value: torch.Tensor) -> torch.Tensor:
        return torch.log(
            value.clamp(min=_EVIDENCE_FLOOR, max=1.0)
        ) / log_scale

    other_direction = F.normalize(
        direction.sum(dim=1, keepdim=True) - direction,
        dim=-1,
        eps=1e-12,
    )
    consensus = (direction * other_direction).sum(dim=-1).clamp(-1.0, 1.0)
    evidence = torch.stack(
        (
            bounded_log_share(group_share)[..., None].expand_as(token_energy),
            bounded_log_share(frequency_share),
            consensus,
        ),
        dim=-1,
    )
    return direction.to(physical_traces.dtype), evidence.to(physical_traces.dtype)


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
        return value + self.ffn(value)


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
    """Read factorized unordered K4 traces and emit the public PI05 LoRA."""

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
        self.evidence_key = torch.nn.Linear(3, width, bias=False)
        self.direction_value = torch.nn.Linear(width, width, bias=False)
        self.physical_value = torch.nn.Linear(width, width, bias=False)
        self.value_fusion = torch.nn.Linear(2 * width, width, bias=False)
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
        physical = video_traces.reshape(
            conditions,
            shots,
            len(self.groups),
            self.temporal_terms,
            self.width,
        )
        direction, evidence = factorize_trace_evidence(physical)
        traces = physical.permute(0, 2, 1, 3, 4).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.width,
        )
        directions = direction.permute(0, 2, 1, 3, 4).reshape_as(traces)
        evidence = evidence.permute(0, 2, 1, 3, 4).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            3,
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
        normalized_directions = self.trace_norm(directions)
        k = self.key(
            normalized_directions + key_route + self.evidence_key(evidence)
        ).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        direction_v = self.direction_value(directions).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        physical_v = self.physical_value(traces).reshape(
            conditions * len(self.groups),
            shots * self.temporal_terms,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        v = torch.cat((direction_v, physical_v), dim=-1)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(
            conditions * len(self.groups),
            self.memory_slots,
            2 * self.width,
        )
        attended = self.value_fusion(attended).reshape(
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


class FixedSemanticExpertRouter(torch.nn.Module):
    """Select fixed top-k parameter owners from a frozen semantic address."""

    def __init__(
        self,
        centers: torch.Tensor,
        *,
        anchor_mean: torch.Tensor,
        top_k: int,
    ) -> None:
        super().__init__()
        if (
            centers.ndim != 2
            or min(centers.shape) <= 0
            or top_k <= 0
            or top_k >= centers.shape[0]
            or anchor_mean.shape != (centers.shape[1],)
            or not bool(torch.isfinite(centers).all())
            or not bool(torch.isfinite(anchor_mean).all())
        ):
            raise FewShotM2PError("invalid fixed semantic expert route")
        self.expert_count = int(centers.shape[0])
        self.anchor_width = int(centers.shape[1])
        self.top_k = int(top_k)
        self.register_buffer(
            "centers",
            F.normalize(centers.to(torch.float32), dim=-1).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "anchor_mean",
            anchor_mean.to(torch.float32).contiguous(),
            persistent=True,
        )

    def forward(self, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor.ndim != 2 or anchor.shape[-1] != self.anchor_width:
            raise FewShotM2PError("fixed semantic route lost task anchor")
        normalized = F.normalize(
            anchor.to(torch.float32) - self.anchor_mean,
            dim=-1,
        )
        indices = torch.topk(
            normalized @ self.centers.transpose(0, 1),
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        weights = torch.full(
            indices.shape,
            1.0 / self.top_k,
            dtype=torch.float32,
            device=anchor.device,
        )
        return indices, weights


class GroundedVideoPolicyLayerTraceM2P(torch.nn.Module):
    """Route grounded K4 video semantics into complete trace experts."""

    def __init__(
        self,
        groups: tuple[PolicyLayerGroup, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        route_centers: torch.Tensor,
        route_anchor_mean: torch.Tensor,
        expert_count: int,
        top_k: int,
        width: int,
        memory_slots: int,
        temporal_terms: int,
        heads: int,
        blocks: int,
        ffn_expansion: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if route_centers.shape[0] != expert_count:
            raise FewShotM2PError("semantic expert count changed")
        self.groups = groups
        self.width = int(width)
        self.memory_slots = int(memory_slots)
        self.temporal_terms = int(temporal_terms)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.router = FixedSemanticExpertRouter(
            route_centers,
            anchor_mean=route_anchor_mean,
            top_k=top_k,
        )
        self.experts = torch.nn.ModuleList(
            PolicyLayerTraceM2P(
                groups,
                template_state=template_state,
                width=width,
                memory_slots=memory_slots,
                temporal_terms=temporal_terms,
                heads=heads,
                blocks=blocks,
                ffn_expansion=ffn_expansion,
                initialization_seed=initialization_seed + expert_index,
            )
            for expert_index in range(expert_count)
        )

    def route(self, grounded_address: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.router(grounded_address)

    def encode(
        self,
        video_traces: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        grounded_address: torch.Tensor,
    ) -> torch.Tensor:
        if (
            condition_video_offsets.ndim != 1
            or condition_video_offsets.dtype != torch.long
            or condition_video_offsets.numel() < 2
        ):
            raise FewShotM2PError("invalid grounded-video expert condition offsets")
        conditions = condition_video_offsets.numel() - 1
        offsets = condition_video_offsets.detach().cpu()
        shot_counts = offsets[1:] - offsets[:-1]
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != video_traces.shape[0]
            or not bool((shot_counts == shot_counts[0]).all())
            or int(shot_counts[0]) <= 1
            or grounded_address.shape != (conditions, self.router.anchor_width)
        ):
            raise FewShotM2PError("grounded-video expert batch changed")
        shots = int(shot_counts[0])
        expected_trace_shape = (
            conditions,
            shots,
            len(self.groups),
            self.temporal_terms,
            self.width,
        )
        if video_traces.shape != (
            conditions * shots,
            len(self.groups),
            self.temporal_terms,
            self.width,
        ):
            raise FewShotM2PError("grounded-video expert traces changed shape")
        physical = video_traces.reshape(expected_trace_shape)
        route_indices, route_weights = self.route(grounded_address)
        memory: torch.Tensor | None = None
        for expert_index, expert in enumerate(self.experts):
            selected = torch.nonzero(
                route_indices == expert_index,
                as_tuple=False,
            )
            if selected.numel() == 0:
                continue
            condition_ids = selected[:, 0]
            route_slots = selected[:, 1]
            local_traces = physical.index_select(0, condition_ids).reshape(
                -1,
                len(self.groups),
                self.temporal_terms,
                self.width,
            )
            local_offsets = torch.arange(
                0,
                (condition_ids.numel() + 1) * shots,
                shots,
                dtype=torch.long,
                device=video_traces.device,
            )
            local_memory = expert.encode(local_traces, local_offsets)
            local_weights = route_weights[condition_ids, route_slots].to(
                dtype=local_memory.dtype
            ).reshape(-1, 1, 1, 1)
            if memory is None:
                memory = local_memory.new_zeros(
                    conditions,
                    len(self.groups),
                    self.memory_slots,
                    self.width,
                )
            memory = memory.index_add(
                0,
                condition_ids,
                local_memory * local_weights,
            )
        expected_memory_shape = (
            conditions,
            len(self.groups),
            self.memory_slots,
            self.width,
        )
        if memory is None or memory.shape != expected_memory_shape or not bool(
            torch.isfinite(memory).all()
        ):
            raise FewShotM2PError("grounded-video expert memory changed shape")
        return memory

    def decode(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.experts[0].decode(memory)

    def forward(
        self,
        video_traces: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        grounded_address: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.decode(
            self.encode(video_traces, condition_video_offsets, grounded_address)
        )


def count_parameters(module: torch.nn.Module) -> int:
    """Return trainable parameter count for the architecture contract."""

    return sum(value.numel() for value in module.parameters() if value.requires_grad)
