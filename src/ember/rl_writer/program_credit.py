"""Direct antithetic closed-loop credit for a Writer policy program."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ember.pi05_source_checkpoint import write_json_atomic
from ember.reward.protocol import RewardProtocolError


@dataclass(frozen=True)
class PairCredit:
    value: float
    mode: str


def program_direction_seed(
    seed_root: int,
    *,
    cycle: int,
    global_task_id: int,
    pair_index: int,
) -> int:
    if min(seed_root, cycle, global_task_id, pair_index) < 0 or pair_index >= 2:
        raise RewardProtocolError("invalid antithetic program direction key")
    payload = (
        f"ember-program-credit-v1|{seed_root}|{cycle}|"
        f"{global_task_id}|{pair_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def program_direction(
    seed: int,
    shape: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    normalized = tuple(int(value) for value in shape)
    if seed < 0 or not normalized or any(value <= 0 for value in normalized):
        raise RewardProtocolError("invalid antithetic program direction shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randint(
        0,
        2,
        normalized,
        generator=generator,
        dtype=torch.int8,
    ).to(torch.float32)
    direction.mul_(2).sub_(1)
    return direction.to(device=device, dtype=dtype, non_blocking=True)


def binary_first_pair_credit(
    *,
    success_plus: bool,
    success_minus: bool,
    progress_plus: float,
    progress_minus: float,
) -> PairCredit:
    values = (float(progress_plus), float(progress_minus))
    if not all(math.isfinite(value) and -1.000001 <= value <= 1.000001 for value in values):
        raise RewardProtocolError("non-finite or unbounded program progress credit")
    if success_plus != success_minus:
        return PairCredit(
            value=1.0 if success_plus else -1.0,
            mode="binary_discordant",
        )
    if success_plus:
        return PairCredit(value=0.0, mode="paired_success_zero")
    return PairCredit(
        value=(values[0] - values[1]) / 2.0,
        mode="paired_failure_semantic",
    )


def program_cotangent(
    directions: Sequence[torch.Tensor],
    credits: Sequence[float],
) -> torch.Tensor:
    if len(directions) != 2 or len(credits) != 2:
        raise RewardProtocolError("program credit requires exactly two antithetic pairs")
    shape = directions[0].shape
    if (
        not shape
        or any(value.shape != shape for value in directions)
        or any(not math.isfinite(float(value)) for value in credits)
        or any(not bool(torch.isfinite(value).all()) for value in directions)
    ):
        raise RewardProtocolError("invalid antithetic program cotangent inputs")
    result = sum(
        direction * float(credit)
        for direction, credit in zip(directions, credits, strict=True)
    )
    result = result / (2.0 * math.sqrt(result.numel()))
    if not bool(torch.isfinite(result).all()):
        raise RewardProtocolError("non-finite antithetic program cotangent")
    return result


def write_program_credit_once(
    *,
    output_dir: Path,
    producer_rank: int,
    cycle: int,
    global_task_id: int,
    teacher_demo_index: int,
    sigma: float,
    pairs: Sequence[dict[str, object]],
    cotangent_norm: float,
) -> None:
    if len(pairs) != 2 or sigma <= 0 or not math.isfinite(cotangent_norm):
        raise RewardProtocolError("invalid program-credit ledger row")
    path = (
        output_dir
        / "program_credit"
        / f"cycle_{cycle:08d}"
        / f"task_{global_task_id:03d}.json"
    )
    if path.exists():
        raise RewardProtocolError(f"program-credit ledger row exists: {path}")
    write_json_atomic(
        path,
        {
            "schema_version": "ember_pi05_antithetic_program_credit_task_v1",
            "producer_rank": producer_rank,
            "outer_cycle": cycle,
            "global_task_id": global_task_id,
            "teacher_demo_index": teacher_demo_index,
            "program_sigma": sigma,
            "program_shape": [320, 256],
            "pairs": list(pairs),
            "cotangent_norm": cotangent_norm,
        },
    )
