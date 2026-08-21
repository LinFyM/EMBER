"""Stable tensor ownership for the EMBER-ECP compiler."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ember.lora import LoRAContract
from ember.pi05_lora import pi05_target_names


ACTION_LAYERS = 18
ACTION_HORIZON = 50
PADDED_ACTION_DIM = 32
TARGET_OWNERS = 38


class TargetFamily(StrEnum):
    Q = "q"
    V = "v"
    ACTION_IN = "action_in"
    ACTION_OUT = "action_out"


@dataclass(frozen=True)
class TargetOwner:
    """One compiler owner aligned exactly with one deployed LoRA target."""

    index: int
    target_name: str
    family: TargetFamily
    layer: int | None
    in_features: int
    out_features: int


_LAYER_TARGET = re.compile(
    r"\.layers\.(?P<layer>\d+)\.self_attn\.(?P<projection>[qv])_proj$"
)


def build_target_owners(contract: LoRAContract) -> tuple[TargetOwner, ...]:
    """Derive the 38 owner rows without inventing a second target ordering."""

    names = tuple(target.name for target in contract.targets)
    if names != pi05_target_names():
        raise ValueError("ECP requires the canonical 38-target PI0.5 LoRA contract")

    owners: list[TargetOwner] = []
    for index, target in enumerate(contract.targets):
        match = _LAYER_TARGET.search(target.name)
        if match is not None:
            layer = int(match.group("layer"))
            family = (
                TargetFamily.Q
                if match.group("projection") == "q"
                else TargetFamily.V
            )
        elif target.name.endswith(".action_in_proj"):
            layer = None
            family = TargetFamily.ACTION_IN
        elif target.name.endswith(".action_out_proj"):
            layer = None
            family = TargetFamily.ACTION_OUT
        else:
            raise ValueError(f"unsupported ECP target: {target.name}")
        owners.append(
            TargetOwner(
                index=index,
                target_name=target.name,
                family=family,
                layer=layer,
                in_features=target.in_features,
                out_features=target.out_features,
            )
        )
    return tuple(owners)
