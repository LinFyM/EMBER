"""Shared layer/rank readout from a policy program to one complete LoRA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch


class LoRAMapperError(RuntimeError):
    """Raised when the public PI05 LoRA topology changes."""


@dataclass(frozen=True)
class LoraTensorSpec:
    """One row-oriented output tensor in a PEFT LoRA state."""

    name: str
    module: str
    module_index: int
    factor_index: int
    rank: int
    width: int
    transpose_output: bool


def build_lora_tensor_specs(
    state: Mapping[str, torch.Tensor],
) -> tuple[LoraTensorSpec, ...]:
    """Build paired A/B output specifications from a real PEFT state."""

    marker_a = ".lora_A.default.weight"
    marker_b = ".lora_B.default.weight"
    modules: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, value in state.items():
        if name.endswith(marker_a):
            marker, factor = marker_a, "A"
        elif name.endswith(marker_b):
            marker, factor = marker_b, "B"
        else:
            raise LoRAMapperError(f"non-LoRA tensor in template: {name}")
        if value.ndim != 2:
            raise LoRAMapperError(f"LoRA tensor is not a matrix: {name}")
        modules.setdefault(name[: -len(marker)], {})[factor] = (name, value)
    if not modules or any(set(pair) != {"A", "B"} for pair in modules.values()):
        raise LoRAMapperError("every target module must contain one LoRA A/B pair")

    result: list[LoraTensorSpec] = []
    for module_index, module in enumerate(sorted(modules)):
        name_a, value_a = modules[module]["A"]
        name_b, value_b = modules[module]["B"]
        rank, input_width = value_a.shape
        output_width, rank_b = value_b.shape
        if rank <= 0 or rank_b != rank:
            raise LoRAMapperError(f"LoRA rank differs for {module}")
        result.extend(
            (
                LoraTensorSpec(
                    name_a,
                    module,
                    module_index,
                    0,
                    rank,
                    input_width,
                    False,
                ),
                LoraTensorSpec(
                    name_b,
                    module,
                    module_index,
                    1,
                    rank,
                    output_width,
                    True,
                ),
            )
        )
    return tuple(result)


class ShapeFamilyMapper(torch.nn.Module):
    """Share one nonlinear family basis while keeping A/B initialization distinct."""

    def __init__(
        self,
        *,
        input_width: int,
        hidden_width: int,
        a_width: int,
        b_width: int,
    ) -> None:
        super().__init__()
        if min(input_width, hidden_width, a_width, b_width) <= 0:
            raise LoRAMapperError("invalid shape-family mapper dimensions")
        self.hidden = torch.nn.Linear(input_width, hidden_width, bias=False)
        self.a = torch.nn.Linear(hidden_width, a_width, bias=False)
        self.b = torch.nn.Linear(hidden_width, b_width, bias=False)
        torch.nn.init.zeros_(self.b.weight)

    def forward(
        self,
        value: torch.Tensor,
        *,
        dynamic_a: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.nn.functional.gelu(self.hidden(value))
        a = self.a(hidden) if dynamic_a else torch.zeros(
            *hidden.shape[:-1],
            self.a.out_features,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return a, self.b(hidden)


class CompleteLoRAMapper(torch.nn.Module):
    """Decode a ``[batch,20,rank,width]`` program into all 76 PI05 tensors."""

    EXPERT_LAYERS = 18
    POLICY_GROUPS = 20
    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )
    _FAMILY_SHAPES = {
        "q": (1024, 2048),
        "v": (1024, 256),
        "action_in": (32, 1024),
        "action_out": (1024, 32),
    }

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        program_width: int,
        mapper_width: int,
        dynamic_a: bool = False,
    ) -> None:
        super().__init__()
        if not tensor_specs or set(template_state) != {item.name for item in tensor_specs}:
            raise LoRAMapperError("Writer LoRA template names changed")
        ranks = {item.rank for item in tensor_specs}
        if len(ranks) != 1:
            raise LoRAMapperError("public Writer LoRA rank is not uniform")
        self.rank = ranks.pop()
        self.program_width = int(program_width)
        self.dynamic_a = bool(dynamic_a)
        self.project = torch.nn.Linear(program_width, mapper_width, bias=False)
        self.families = torch.nn.ModuleDict(
            {
                family: ShapeFamilyMapper(
                    input_width=mapper_width,
                    hidden_width=mapper_width,
                    a_width=shapes[0],
                    b_width=shapes[1],
                )
                for family, shapes in self._FAMILY_SHAPES.items()
            }
        )
        self.tensor_specs = tensor_specs
        self._template_buffers: dict[str, str] = {}
        self._decoding: dict[str, tuple[str, int, int]] = {}
        observed: set[tuple[str, int]] = set()
        for index, item in enumerate(tensor_specs):
            family, group = self._decode_owner(item)
            expected = self._FAMILY_SHAPES[family][item.factor_index]
            if item.width != expected:
                raise LoRAMapperError("sealed PI05 LoRA target width changed")
            observed.add((family, group))
            template = template_state[item.name].detach().contiguous()
            if item.factor_index == 1 and torch.count_nonzero(template):
                raise LoRAMapperError("LoRA-B template must begin at physical zero")
            buffer = f"template_{index:03d}"
            self.register_buffer(buffer, template, persistent=True)
            self._template_buffers[item.name] = buffer
            self._decoding[item.name] = (family, group, item.factor_index)
        expected_owners = {
            *((family, layer + 1) for layer in range(self.EXPERT_LAYERS) for family in ("q", "v")),
            ("action_in", 0),
            ("action_out", self.POLICY_GROUPS - 1),
        }
        if observed != expected_owners:
            raise LoRAMapperError("sealed PI05 LoRA target ownership changed")

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int]:
        if item.module.endswith("action_in_proj"):
            return "action_in", 0
        if item.module.endswith("action_out_proj"):
            return "action_out", self.POLICY_GROUPS - 1
        match = self._EXPERT_MODULE.fullmatch(item.module)
        if match is None:
            raise LoRAMapperError(f"unsupported PI05 task-LoRA module: {item.module}")
        layer = int(match.group(1))
        if not 0 <= layer < self.EXPERT_LAYERS:
            raise LoRAMapperError("PI05 task-LoRA layer is outside Action Expert")
        return match.group(2)[0], layer + 1

    def template_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer)
            for name, buffer in self._template_buffers.items()
        }

    def forward(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (self.POLICY_GROUPS, self.rank, self.program_width)
        if program.ndim != 4 or tuple(program.shape[1:]) != expected:
            raise LoRAMapperError("layer/rank program topology changed")
        projected = self.project(program)
        family_outputs: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for family, mapper in self.families.items():
            groups = (
                range(1, self.EXPERT_LAYERS + 1)
                if family in {"q", "v"}
                else (0,) if family == "action_in" else (self.POLICY_GROUPS - 1,)
            )
            for group in groups:
                family_outputs[(family, group)] = mapper(
                    projected[:, group], dynamic_a=self.dynamic_a
                )

        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            family, group, factor = self._decoding[item.name]
            rows = family_outputs[(family, group)][factor]
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(template.dtype) + template[None]
            result[item.name] = value[0] if program.shape[0] == 1 else value
        return result
