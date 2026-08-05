"""Video-owned invariant program and policy-wide target-token LoRA decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class FewShotM2PError(RuntimeError):
    """Raised when the K-shot program or policy-target topology changes."""


class ZeroPreservingBlock(torch.nn.Module):
    """Bias-free pre-norm self-attention block that maps exact zero to zero."""

    def __init__(self, width: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        if min(width, heads, expansion) <= 0 or width % heads:
            raise FewShotM2PError("invalid zero-preserving transformer block")
        self.norm_attention = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            bias=False,
            batch_first=True,
        )
        self.norm_ffn = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        value = value + attended
        return value + self.ffn(self.norm_ffn(value))


class VideoValueCrossAttention(torch.nn.Module):
    """Cross-attend with routing-only queries while content comes only from values."""

    def __init__(self, query_width: int, value_width: int, width: int, heads: int) -> None:
        super().__init__()
        if min(query_width, value_width, width, heads) <= 0 or width % heads:
            raise FewShotM2PError("invalid video-value cross attention")
        self.heads = int(heads)
        self.head_width = width // heads
        self.query_norm = torch.nn.LayerNorm(query_width, elementwise_affine=False)
        self.value_norm = torch.nn.LayerNorm(value_width, elementwise_affine=False)
        self.query = torch.nn.Linear(query_width, width, bias=False)
        self.key = torch.nn.Linear(value_width, width, bias=False)
        self.value = torch.nn.Linear(value_width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key_route: torch.Tensor,
        values: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            query.ndim != 3
            or key_route.shape != values.shape
            or values.ndim != 3
            or query.shape[0] != values.shape[0]
            or (value_mask is not None and value_mask.shape != values.shape[:2])
        ):
            raise FewShotM2PError("invalid video-value attention batch")
        batch, query_count = query.shape[:2]
        value_count = values.shape[1]
        q = self.query(self.query_norm(query)).reshape(
            batch, query_count, self.heads, self.head_width
        ).transpose(1, 2)
        normalized_values = self.value_norm(values)
        k = self.key(normalized_values + key_route).reshape(
            batch, value_count, self.heads, self.head_width
        ).transpose(1, 2)
        v = self.value(values).reshape(
            batch, value_count, self.heads, self.head_width
        ).transpose(1, 2)
        mask = None
        if value_mask is not None:
            mask = torch.zeros(
                batch,
                1,
                query_count,
                value_count,
                dtype=q.dtype,
                device=q.device,
            ).masked_fill(~value_mask[:, None, None], float("-inf"))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        merged = attended.transpose(1, 2).reshape(batch, query_count, -1)
        return self.output(merged)


class InvariantProgramEncoder(torch.nn.Module):
    """Read an unordered K-video set into video-owned invariant program slots."""

    TEMPORAL_TERMS = 4

    def __init__(
        self,
        *,
        task_width: int,
        video_width: int,
        program_width: int,
        program_slots: int,
        heads: int,
        blocks: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(
            task_width,
            video_width,
            program_width,
            program_slots,
            heads,
            blocks,
        ) <= 0:
            raise FewShotM2PError("invalid invariant program topology")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        latent = torch.empty(program_slots, program_width)
        latent.normal_(mean=0.0, std=0.02, generator=generator)
        temporal = torch.empty(self.TEMPORAL_TERMS, program_width)
        temporal.normal_(mean=0.0, std=0.02, generator=generator)
        self.latent_route = torch.nn.Parameter(latent)
        self.temporal_route = torch.nn.Parameter(temporal)
        self.task_route = torch.nn.Linear(task_width, program_width, bias=False)
        self.video_route = torch.nn.Linear(video_width, program_width, bias=False)
        self.reader = VideoValueCrossAttention(
            query_width=program_width,
            value_width=program_width,
            width=program_width,
            heads=heads,
        )
        self.blocks = torch.nn.ModuleList(
            ZeroPreservingBlock(program_width, heads) for _ in range(blocks)
        )
        self.program_slots = int(program_slots)
        self.program_width = int(program_width)

    def forward(
        self,
        task_descriptors: torch.Tensor,
        video_tokens: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_descriptors.ndim != 2
            or video_tokens.ndim != 3
            or video_tokens.shape[0] != task_descriptors.shape[0]
            or video_tokens.shape[1] != self.TEMPORAL_TERMS
            or condition_video_offsets.ndim != 1
            or condition_video_offsets.dtype != torch.long
            or condition_video_offsets.numel() < 2
        ):
            raise FewShotM2PError("invalid few-shot condition descriptors")
        offsets = tuple(int(value) for value in condition_video_offsets.cpu().tolist())
        if (
            offsets[0] != 0
            or offsets[-1] != video_tokens.shape[0]
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise FewShotM2PError("invalid condition-video offsets")
        shot_counts = {right - left for left, right in zip(offsets, offsets[1:])}
        if len(shot_counts) != 1:
            raise FewShotM2PError("few-shot conditions must use one fixed K")
        shots = shot_counts.pop()
        conditions = len(offsets) - 1
        if shots <= 1:
            raise FewShotM2PError("invariant program requires more than one video")

        projected = self.video_route(video_tokens)
        projected = projected.reshape(
            conditions,
            shots * self.TEMPORAL_TERMS,
            self.program_width,
        )
        temporal_route = self.temporal_route.repeat(shots, 1)
        key_route = temporal_route[None].expand(conditions, -1, -1)
        task_by_condition = torch.stack(
            [task_descriptors[left:right].mean(dim=0) for left, right in zip(offsets, offsets[1:])]
        )
        query = self.latent_route[None] + self.task_route(task_by_condition)[:, None]
        content = self.reader(query, key_route, projected)
        for block in self.blocks:
            content = block(content)
        return content


@dataclass(frozen=True)
class PolicyTargetSpec:
    """One PI05 module with paired A/B output tensors."""

    module_index: int
    module: str
    a_name: str
    b_name: str
    rank: int
    input_width: int
    output_width: int
    a_transpose: bool
    b_transpose: bool


def build_policy_target_specs(tensor_specs: Sequence[object]) -> tuple[PolicyTargetSpec, ...]:
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
                a_transpose=bool(getattr(a, "transpose_output")),
                b_transpose=bool(getattr(b, "transpose_output")),
            )
        )
    return tuple(result)


class TargetFactorHead(torch.nn.Module):
    """Bias-free target-owned row head with exact-zero final initialization."""

    def __init__(self, width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(width, hidden_width, bias=False)
        self.output = torch.nn.Linear(hidden_width, output_width, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(value)))


class PolicyM2PDecoder(torch.nn.Module):
    """Jointly decode all target/layer/rank tokens into one complete PI05 LoRA."""

    def __init__(
        self,
        targets: tuple[PolicyTargetSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        program_width: int,
        heads: int,
        blocks: int,
        head_hidden_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not targets or min(program_width, heads, blocks, head_hidden_width) <= 0:
            raise FewShotM2PError("invalid policy M2P topology")
        ranks = {item.rank for item in targets}
        if len(ranks) != 1:
            raise FewShotM2PError("public policy target ranks changed")
        self.rank = ranks.pop()
        self.targets = targets
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def route(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, program_width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.target_route = route(len(targets))
        self.rank_route = route(self.rank)
        self.query_route = route(len(targets) * self.rank)
        self.initial_reader = VideoValueCrossAttention(
            query_width=program_width,
            value_width=program_width,
            width=program_width,
            heads=heads,
        )
        self.blocks = torch.nn.ModuleList(
            ZeroPreservingBlock(program_width, heads) for _ in range(blocks)
        )
        self.refine_query = torch.nn.Linear(program_width, program_width, bias=False)
        self.refine_reader = VideoValueCrossAttention(
            query_width=program_width,
            value_width=program_width,
            width=program_width,
            heads=heads,
        )
        self.a_heads = torch.nn.ModuleDict()
        self.b_heads = torch.nn.ModuleDict()
        self._template_names: dict[str, str] = {}
        for index, target in enumerate(targets):
            key = f"target_{index:03d}"
            self.a_heads[key] = TargetFactorHead(
                program_width, head_hidden_width, target.input_width
            )
            self.b_heads[key] = TargetFactorHead(
                program_width, head_hidden_width, target.output_width
            )
            for factor, name in (("a", target.a_name), ("b", target.b_name)):
                if name not in template_state:
                    raise FewShotM2PError("policy M2P template tensor missing")
                buffer_name = f"template_{index:03d}_{factor}"
                self.register_buffer(
                    buffer_name,
                    template_state[name].detach().contiguous(),
                    persistent=True,
                )
                self._template_names[name] = buffer_name

    def _routing(self) -> torch.Tensor:
        structured = self.target_route[:, None] + self.rank_route[None]
        return structured.reshape(-1, structured.shape[-1]) + self.query_route

    def forward(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        if program.ndim != 3:
            raise FewShotM2PError("policy M2P program changed shape")
        routing = self._routing()[None].expand(program.shape[0], -1, -1)
        zero_route = torch.zeros_like(program)
        content = self.initial_reader(routing, zero_route, program)
        for block in self.blocks:
            content = block(content)
        refined = self.refine_reader(
            self.refine_query(content) + routing,
            zero_route,
            program,
        )
        content = content + refined
        content = content.reshape(
            program.shape[0], len(self.targets), self.rank, -1
        )
        result: dict[str, torch.Tensor] = {}
        for index, target in enumerate(self.targets):
            key = f"target_{index:03d}"
            token = content[:, index]
            a = self.a_heads[key](token)
            b = self.b_heads[key](token)
            if target.a_transpose:
                a = a.transpose(-1, -2)
            if target.b_transpose:
                b = b.transpose(-1, -2)
            template_a = getattr(self, self._template_names[target.a_name])
            template_b = getattr(self, self._template_names[target.b_name])
            result[target.a_name] = a.to(template_a.dtype) + template_a[None]
            result[target.b_name] = b.to(template_b.dtype) + template_b[None]
        return result


def count_parameters(module: torch.nn.Module) -> int:
    """Return trainable parameter count for the architecture contract."""

    return sum(value.numel() for value in module.parameters() if value.requires_grad)
