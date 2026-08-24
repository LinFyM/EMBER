"""Independent action-query sampling and functional sensitivity for G1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from ember.ecp.contracts import TargetOwner
from ember.ecp.g1_assets import G1RankAssets, G1TaskAssets
from ember.ecp.g1_objective import family_balanced_sensitivity_weights
from ember.ecp.native_materialization import compose_rank12_plus_rank4
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.writer.data import FunctionalQueryDataset
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
)


def policy_rng_seed(config: Mapping[str, Any], task: G1TaskAssets, step: int) -> int:
    return (
        int(config["functional_query"]["sampling_seed"])
        + task.ordinal * 1_000_003
        + int(step)
    )


def functional_batch(
    *,
    dataset: FunctionalQueryDataset,
    processor: Pi05LiberoProcessor,
    task: G1TaskAssets,
    config: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    demos = tuple(map(int, config["functional_query"]["demo_indices"]))
    rows_by_demo = dataset.task_episode_rows[task.global_task_id]
    if set(rows_by_demo) != set(demos):
        raise ValueError("G1 functional-query episodes changed")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(config["functional_query"]["sampling_seed"]),
                task.ordinal,
                int(step),
                0xF17E,
            ]
        )
    )
    selected = [
        rows[int(generator.integers(0, len(rows)))]
        for demo in demos
        for rows in (rows_by_demo[demo],)
    ]
    if len(selected) != int(config["functional_query"]["batch_size"]):
        raise ValueError("G1 functional logical batch changed")
    return processor.training_batch(
        default_collate([dataset[index] for index in selected])
    )


def functional_gradient(
    *,
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    task: G1TaskAssets,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, details, gradients = functional_lora_loss_gradient(
        policy,
        state,
        contract,
        batch=batch,
        policy_rng_seed=policy_rng_seed(config, task, step),
        policy_rng_device=next(policy.parameters()).device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            config["functional_query"]["policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    if details or not torch.isfinite(loss):
        raise ValueError("G1 independent functional loss changed")
    return loss.float(), gradients


def gradient_bridge(
    loss: torch.Tensor,
    state: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    bridge = loss.detach().clone()
    for name, value in state.items():
        bridge = bridge + ((value - value.detach()) * gradients[name].to(value)).sum()
    return bridge


def _reference_probe_state(
    *,
    reference: Mapping[str, torch.Tensor],
    carrier_rank12: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    zero_b = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        zero_b[a_name] = reference[a_name]
        zero_b[b_name] = torch.zeros_like(reference[b_name])
    return compose_rank12_plus_rank4(
        carrier_state=carrier_rank12,
        residual_state=zero_b,
        rank16_contract=contract,
    )


def calibrate_policy_sensitivity(
    *,
    policy: torch.nn.Module,
    ranks: G1RankAssets,
    references: Sequence[Mapping[str, torch.Tensor]],
    owners: tuple[TargetOwner, ...],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    task: G1TaskAssets,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for member, reference in enumerate(references):
        probe_state = _reference_probe_state(
            reference=reference,
            carrier_rank12=ranks.carrier_rank12,
            contract=ranks.contract,
        )
        _, gradients = functional_gradient(
            policy=policy,
            state=probe_state,
            contract=ranks.contract,
            batch=batch,
            config=config,
            task=task,
            step=0x5100 + member,
        )
        per_target = []
        for target in ranks.contract.targets:
            b_name = target.name + LORA_B_SUFFIX
            directional = (
                gradients[b_name][:, 12:].float() * reference[b_name].float()
            ).sum()
            per_target.append(directional.abs())
        rows.append(torch.stack(per_target))
    raw = torch.stack(rows)
    weights = family_balanced_sensitivity_weights(raw, owners)
    if not torch.isfinite(weights).all() or not torch.allclose(
        weights.sum(1), torch.ones(len(references), device=weights.device)
    ):
        raise ValueError("G1 policy-sensitivity calibration is invalid")
    return raw.detach(), weights.detach()
