"""Target-family-aware compiler from an ECP Program to one complete PI0.5 LoRA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.program import ECPProgram
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)


# 0.4 x the per-family canonical expert factor RMS divided by sqrt(width=256).
DEFAULT_ABSOLUTE_HEAD_INIT = {
    "action_in": {"a": 7.972e-4, "b": 1.409e-4},
    "action_out": {"a": 7.540e-5, "b": 4.268e-4},
    "q": {"a": 2.070e-4, "b": 1.464e-4},
    "v": {"a": 1.294e-4, "b": 2.588e-4},
}


@dataclass(frozen=True)
class ECPCompilerOutput:
    state: Mapping[str, torch.Tensor]
    locality_penalty: torch.Tensor
    exact_owner_attention: torch.Tensor


def select_compiled_state(
    state: Mapping[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in state.items()}


class TargetFamilyCompiler(torch.nn.Module):
    """Use numeric target/rank queries without collapsing Program ownership."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        *,
        program_width: int = 128,
        compiler_width: int = 256,
        event_slots: int = 8,
        factor_head_init: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__()
        validate_lora_state(template_state, contract)
        self.owners = owners
        self.contract = contract
        self.owner_count = len(owners)
        self.rank = int(contract.rank)
        self.event_slots = event_slots
        self.compiler_width = compiler_width
        self.language_projection = torch.nn.Linear(
            program_width, compiler_width, bias=False
        )
        self.scene_projection = torch.nn.Linear(
            program_width, compiler_width, bias=False
        )
        self.process_projection = torch.nn.Linear(
            program_width, compiler_width, bias=False
        )
        self.uncertainty_projection = torch.nn.Linear(
            program_width, compiler_width, bias=False
        )
        self.program_owner_embedding = torch.nn.Embedding(
            self.owner_count, compiler_width
        )
        self.target_embedding = torch.nn.Embedding(self.owner_count, compiler_width)
        self.rank_embedding = torch.nn.Embedding(self.rank, compiler_width)
        self.family_embedding = torch.nn.Embedding(
            len(TargetFamily), compiler_width
        )
        self.layer_embedding = torch.nn.Embedding(18, compiler_width)
        self.token_type_embedding = torch.nn.Embedding(3, compiler_width)
        self.event_embedding = torch.nn.Embedding(event_slots, compiler_width)
        self.query_projection = torch.nn.Linear(
            compiler_width, compiler_width, bias=False
        )
        self.key_projection = torch.nn.Linear(
            compiler_width, compiler_width, bias=False
        )
        self.value_projection = torch.nn.Linear(
            compiler_width, compiler_width, bias=False
        )
        self.query_content_modulation = torch.nn.Linear(
            compiler_width, compiler_width, bias=False
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(compiler_width),
            torch.nn.Linear(compiler_width, 2 * compiler_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(2 * compiler_width, compiler_width, bias=False),
            torch.nn.LayerNorm(compiler_width),
        )
        self.factor_a, self.factor_b = self._factor_heads(owners, compiler_width)
        self._register_coordinates(owners)
        self._register_templates(template_state)
        init = factor_head_init or DEFAULT_ABSOLUTE_HEAD_INIT
        for family, head in self.factor_a.items():
            torch.nn.init.normal_(head.weight, std=float(init[family]["a"]))
        for family, head in self.factor_b.items():
            torch.nn.init.normal_(head.weight, std=float(init[family]["b"]))

    @staticmethod
    def _factor_heads(
        owners: tuple[TargetOwner, ...], width: int
    ) -> tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]:
        shapes: dict[str, tuple[int, int]] = {}
        for owner in owners:
            shapes.setdefault(
                owner.family.value, (owner.in_features, owner.out_features)
            )
        return (
            torch.nn.ModuleDict(
                {
                    family: torch.nn.Linear(width, shape[0], bias=False)
                    for family, shape in shapes.items()
                }
            ),
            torch.nn.ModuleDict(
                {
                    family: torch.nn.Linear(width, shape[1], bias=False)
                    for family, shape in shapes.items()
                }
            ),
        )

    def _register_coordinates(self, owners: tuple[TargetOwner, ...]) -> None:
        families = tuple(TargetFamily)
        family_ids = torch.tensor(
            [families.index(owner.family) for owner in owners], dtype=torch.long
        )
        layer_ids = torch.tensor(
            [
                owner.layer
                if owner.layer is not None
                else (17 if owner.family is TargetFamily.ACTION_OUT else 0)
                for owner in owners
            ],
            dtype=torch.long,
        )
        token_owner_ids = torch.cat(
            (
                torch.arange(self.owner_count),
                torch.arange(self.owner_count),
                torch.arange(self.owner_count).repeat(self.event_slots),
            )
        )
        self.register_buffer("family_ids", family_ids, persistent=False)
        self.register_buffer("layer_ids", layer_ids, persistent=False)
        self.register_buffer("token_owner_ids", token_owner_ids, persistent=False)
        source_family = family_ids[token_owner_ids]
        source_layer = layer_ids[token_owner_ids]
        distance = (layer_ids[:, None] - source_layer[None]).abs().float()
        mismatch = (family_ids[:, None] != source_family[None]).float()
        exact = torch.arange(self.owner_count)[:, None] == token_owner_ids[None]
        bias = -0.75 * distance - 1.5 * mismatch + 3.0 * exact.float()
        cost = distance / 17.0 + mismatch
        self.register_buffer("locality_bias", bias, persistent=False)
        self.register_buffer("locality_cost", cost, persistent=False)
        self.register_buffer("exact_owner_mask", exact, persistent=False)

    def _register_templates(self, state: Mapping[str, torch.Tensor]) -> None:
        self._template_names: dict[str, str] = {}
        for index, name in enumerate(sorted(state)):
            buffer = f"template_{index:03d}"
            self.register_buffer(
                buffer, state[name].detach().clone().contiguous(), persistent=True
            )
            self._template_names[name] = buffer

    def template_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer) for name, buffer in self._template_names.items()
        }

    def _tokens(
        self, program: ECPProgram
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = program.language.shape[0]
        expected = (batch, self.owner_count)
        if (
            program.language.shape[:2] != expected
            or program.scene.shape[:2] != expected
            or program.process.shape[:3]
            != (batch, self.event_slots, self.owner_count)
            or program.presence.shape != (batch, self.event_slots)
            or program.uncertainty.shape != program.process.shape
        ):
            raise ValueError("compiler Program tensors changed shape")
        language = torch.nn.functional.layer_norm(
            self.language_projection(program.language),
            (self.compiler_width,),
        )
        scene = torch.nn.functional.layer_norm(
            self.scene_projection(program.scene),
            (self.compiler_width,),
        )
        process = torch.nn.functional.layer_norm(
            self.process_projection(program.process)
            + self.uncertainty_projection(torch.log1p(program.uncertainty.float())),
            (self.compiler_width,),
        )
        value_tokens = torch.cat((language, scene, process.flatten(1, 2)), dim=1)
        owner_bias = self.program_owner_embedding.weight[None]
        language_key = language + owner_bias + self.token_type_embedding.weight[0]
        scene_key = scene + owner_bias + self.token_type_embedding.weight[1]
        process_key = (
            process
            + owner_bias[:, None]
            + self.event_embedding.weight[None, :, None]
            + self.token_type_embedding.weight[2]
        )
        key_tokens = torch.cat(
            (language_key, scene_key, process_key.flatten(1, 2)), dim=1
        )
        process_presence = program.presence[:, :, None].expand(
            -1, -1, self.owner_count
        ).flatten(1)
        presence = torch.cat(
            (
                torch.ones(
                    batch,
                    2 * self.owner_count,
                    device=value_tokens.device,
                    dtype=process_presence.dtype,
                ),
                process_presence,
            ),
            dim=1,
        )
        return key_tokens, value_tokens, presence

    def _queries(self) -> torch.Tensor:
        target = self.target_embedding.weight[:, None]
        rank = self.rank_embedding.weight[None]
        family = self.family_embedding(self.family_ids)[:, None]
        layer = self.layer_embedding(self.layer_ids)[:, None]
        return target + rank + family + layer

    def forward(self, program: ECPProgram) -> ECPCompilerOutput:
        key_tokens, value_tokens, presence = self._tokens(program)
        queries = self.query_projection(self._queries())
        keys = self.key_projection(key_tokens)
        values = self.value_projection(value_tokens)
        logits = torch.einsum("jrd,bnd->bjrn", queries, keys) / math.sqrt(
            self.compiler_width
        )
        logits = logits + self.locality_bias[:, None, :]
        logits = logits + presence.clamp_min(1e-4).log()[:, None, None]
        attention = logits.softmax(-1)
        hidden = torch.einsum("bjrn,bnd->bjrd", attention, values)
        modulation = 1.0 + torch.tanh(self.query_content_modulation(queries))
        hidden = hidden * modulation[None]
        hidden = self.trunk(hidden)
        templates = self.template_state()
        result: dict[str, torch.Tensor] = {}
        for owner in self.owners:
            family = owner.family.value
            addressed = hidden[:, owner.index]
            full_a = self.factor_a[family](addressed)
            full_b = self.factor_b[family](addressed).transpose(1, 2)
            name_a = owner.target_name + LORA_A_SUFFIX
            name_b = owner.target_name + LORA_B_SUFFIX
            # Prior-only and full Programs must inhabit one learned compiler
            # surface. Template buffers remain initialization/coordinate
            # authorities, but never bypass the direct family heads.
            result[name_a] = full_a.to(templates[name_a])
            result[name_b] = full_b.to(templates[name_b])
        locality = torch.einsum(
            "bjrn,jn->", attention.float(), self.locality_cost
        ) / (attention.shape[0] * self.owner_count * self.rank)
        exact_attention = (attention * self.exact_owner_mask[:, None]).sum(-1).mean()
        return ECPCompilerOutput(
            state=result,
            locality_penalty=locality,
            exact_owner_attention=exact_attention,
        )
