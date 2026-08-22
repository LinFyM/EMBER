"""Fit19-only initialization of layer-resolved compiler heads to the prior."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.distributed as dist

from ember.ecp.compiler import LayerResolvedCompiler
from ember.ecp.program import ECPProgram
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


@torch.no_grad()
def calibrate_prior_heads(
    compiler: LayerResolvedCompiler,
    programs: Mapping[int, ECPProgram],
    *,
    relative_ridge: float,
) -> dict[str, float | int]:
    """Apply a minimum-change target-local ridge correction to the stable prior."""

    if not programs or relative_ridge <= 0:
        raise ValueError("prior head calibration requires Programs and ridge")
    hidden = torch.cat(
        [
            compiler.addressed_hidden(program.prior_only()).float()
            for _, program in sorted(programs.items())
        ],
        dim=0,
    )
    templates = compiler.template_state()
    before_squared = hidden.new_zeros(())
    after_squared = hidden.new_zeros(())
    target_squared = hidden.new_zeros(())
    identity = torch.eye(
        compiler.compiler_width, device=hidden.device, dtype=hidden.dtype
    )
    for owner in compiler.owners:
        key = compiler.owner_head_key(owner)
        x = hidden[:, owner.index].reshape(-1, compiler.compiler_width)
        name_a = owner.target_name + LORA_A_SUFFIX
        name_b = owner.target_name + LORA_B_SUFFIX
        target_a = templates[name_a].float()[None].expand(hidden.shape[0], -1, -1)
        target_b = (
            templates[name_b]
            .float()
            .transpose(0, 1)[None]
            .expand(hidden.shape[0], -1, -1)
        )
        target = torch.cat(
            (
                target_a.reshape(-1, owner.in_features),
                target_b.reshape(-1, owner.out_features),
            ),
            dim=-1,
        )
        weights = torch.cat(
            (
                compiler.factor_a[key].weight.float(),
                compiler.factor_b[key].weight.float(),
            ),
            dim=0,
        )
        prediction = x @ weights.transpose(0, 1)
        residual = target - prediction
        gram = x.transpose(0, 1) @ x
        ridge = relative_ridge * gram.diagonal().mean().clamp_min(1e-8)
        correction = torch.linalg.solve(
            gram + ridge * identity,
            x.transpose(0, 1) @ residual,
        ).transpose(0, 1)
        calibrated = weights + correction
        split = owner.in_features
        compiler.factor_a[key].weight.copy_(calibrated[:split])
        compiler.factor_b[key].weight.copy_(calibrated[split:])
        before_squared += residual.square().sum()
        after_squared += (target - x @ calibrated.transpose(0, 1)).square().sum()
        target_squared += target.square().sum()
    return {
        "fit_programs": len(programs),
        "relative_ridge": float(relative_ridge),
        "relative_residual_before": float(
            (before_squared / target_squared.clamp_min(1e-12)).sqrt()
        ),
        "relative_residual_after": float(
            (after_squared / target_squared.clamp_min(1e-12)).sqrt()
        ),
    }


def calibrate_prior_heads_distributed(
    *,
    compiler: LayerResolvedCompiler,
    programs: Mapping[int, ECPProgram],
    relative_ridge: float,
    context: object,
    resume: bool,
) -> dict[str, object]:
    """Calibrate once on rank zero and synchronize the shared compiler."""

    if resume:
        return {"applied": False, "reason": "resume_restores_mdco_checkpoint"}
    summary: dict[str, object] = {}
    if context.is_main:
        summary = {
            "applied": True,
            **calibrate_prior_heads(
                compiler, programs, relative_ridge=relative_ridge
            ),
            "fit_only": True,
            "held_action_or_reward_reads": 0,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
        }
    if context.world_size > 1:
        for parameter in compiler.parameters():
            dist.broadcast(parameter.data, src=0)
        payload = [summary if context.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        summary = dict(payload[0])
    return summary
