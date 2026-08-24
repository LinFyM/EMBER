#!/usr/bin/env python3
"""Measure the actual G1 video-bank subspaces against successful rank4 members.

This is a read-only mechanism diagnostic.  It captures one held task's frozen
native X/Y stream once, forms centered second moments for the exact input and
output candidate measures, and projects each verified mobile-rank4 member into
those subspaces.  It does not optimize or select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.g1_assets import G1_MEMBER_NAMES
from ember.ecp.g1_runtime import G1_CHECKPOINT_SCHEMA, REPO_ROOT, prepare_runtime
from ember.ecp.native_factors import (
    NativeOutputBankState,
    native_output_group_count,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    small_core_balanced_svd,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic


SCHEMA = "ember_ecp_native_factor_g1_bank_span_v1"


@dataclass
class CenteredMoment:
    """Streaming centered scatter for one candidate-vector measure."""

    width: int
    device: torch.device
    count: int = 0
    mean: torch.Tensor | None = None
    scatter: torch.Tensor | None = None

    def add(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1, self.width).float()
        if flat.shape[0] <= 0:
            raise ValueError("native bank chunk is empty")
        chunk_count = int(flat.shape[0])
        chunk_mean = flat.mean(0)
        centered = flat - chunk_mean
        chunk_scatter = centered.transpose(0, 1) @ centered
        if self.mean is None or self.scatter is None:
            self.count = chunk_count
            self.mean = chunk_mean
            self.scatter = chunk_scatter
            return
        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        correction = float(self.count * chunk_count) / float(total)
        self.scatter = self.scatter + chunk_scatter + correction * torch.outer(
            delta, delta
        )
        self.mean = self.mean + delta * (float(chunk_count) / float(total))
        self.count = total

    def basis(
        self, *, relative_threshold: float
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.mean is None or self.scatter is None or self.count <= 1:
            raise ValueError("native bank moment is incomplete")
        eigenvalues, eigenvectors = torch.linalg.eigh(self.scatter)
        singular = eigenvalues.clamp_min(0).sqrt()
        maximum = singular[-1].clamp_min(torch.finfo(singular.dtype).tiny)
        keep = singular > maximum * relative_threshold
        basis = eigenvectors[:, keep]
        ranks = {
            f"relative_{threshold:g}": int(
                (singular > maximum * threshold).sum().item()
            )
            for threshold in (1e-3, 1e-4, 1e-5, 1e-6)
        }
        return basis, {
            "candidate_count": self.count,
            "width": self.width,
            "projection_relative_singular_threshold": relative_threshold,
            "projection_rank": int(basis.shape[1]),
            "ranks": ranks,
            "maximum_singular_value": float(maximum.detach().cpu()),
            "minimum_retained_singular_ratio": (
                float((singular[keep][0] / maximum).detach().cpu())
                if torch.any(keep)
                else None
            ),
        }


def _energy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((b.transpose(0, 1) @ b) * (a @ a.transpose(0, 1))).sum()


def _distance_squared(
    a: torch.Tensor,
    b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
) -> torch.Tensor:
    return (
        _energy(a, b)
        + _energy(reference_a, reference_b)
        - 2.0
        * ((b.transpose(0, 1) @ reference_b) * (a @ reference_a.transpose(0, 1))).sum()
    ).clamp_min(0)


def _clip_singular_components(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_cap: torch.Tensor,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    canonical_a, canonical_b = small_core_balanced_svd(a, b)
    singular = canonical_a.float().square().sum(-1)
    cap = scale_cap.float() * math.sqrt(in_features * out_features)
    clipped = torch.minimum(singular, cap)
    ratio = torch.where(
        singular > 0,
        (clipped / singular.clamp_min(1e-30)).sqrt(),
        torch.zeros_like(singular),
    )
    return (
        canonical_a * ratio[:, None],
        canonical_b * ratio[None, :],
        (singular / cap.clamp_min(1e-30)).detach().cpu().tolist(),
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    runtime = None
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        runtime = prepare_runtime(args)
        torch.backends.cuda.matmul.allow_tf32 = False
        input_moments = [
            CenteredMoment(owner.in_features, args.torch_device)
            for owner in runtime.owners
        ]
        output_moments = [
            tuple(
                CenteredMoment(
                    owner.out_features // native_output_group_count(owner),
                    args.torch_device,
                )
                for _ in range(native_output_group_count(owner))
            )
            for owner in runtime.owners
        ]
        boundaries = [
            NativeOutputBankState(final=value.detach())
            for value in runtime.video.readout.final_outputs
        ]
        next_frame = 0
        with torch.no_grad():
            for chunk in runtime.video.readout.chunks():
                if chunk.start_frame != next_frame:
                    raise ValueError("native bank diagnostic lost frame order")
                for target, (owner, x, y, input_moment, output_groups, boundary) in enumerate(
                    zip(
                        runtime.owners,
                        chunk.inputs,
                        chunk.outputs,
                        input_moments,
                        output_moments,
                        boundaries,
                        strict=True,
                    )
                ):
                    del target
                    input_moment.add(x)
                    bank = boundary.build(y, start_frame=chunk.start_frame)
                    groups = native_output_group_count(owner)
                    grouped = bank.reshape(
                        *bank.shape[:-1], groups, bank.shape[-1] // groups
                    ).movedim(-2, 0)
                    for group, moment in enumerate(output_groups):
                        moment.add(grouped[group])
                next_frame += chunk.frame_count
        if next_frame != runtime.video.readout.frame_count:
            raise ValueError("native bank diagnostic stream ended early")

        input_bases = []
        output_bases = []
        span_rows = []
        for owner, input_moment, output_group_moments in zip(
            runtime.owners, input_moments, output_moments, strict=True
        ):
            input_basis, input_summary = input_moment.basis(
                relative_threshold=args.relative_singular_threshold
            )
            group_bases = []
            group_summaries = []
            for moment in output_group_moments:
                basis, summary = moment.basis(
                    relative_threshold=args.relative_singular_threshold
                )
                group_bases.append(basis)
                group_summaries.append(summary)
            input_bases.append(input_basis)
            output_bases.append(tuple(group_bases))
            span_rows.append(
                {
                    "target": owner.target_name,
                    "family": owner.family.value,
                    "input": input_summary,
                    "output_groups": group_summaries,
                }
            )

        member_rows = []
        materialized_residual: dict[str, torch.Tensor] = {}
        references = runtime.ranks.reference_rank4[runtime.task.ordinal]
        for member_index, (member_name, reference) in enumerate(
            zip(G1_MEMBER_NAMES, references, strict=True)
        ):
            target_rows = []
            family_totals: dict[str, dict[str, float]] = defaultdict(
                lambda: {
                    "reference_energy": 0.0,
                    "projected_energy": 0.0,
                    "projected_error": 0.0,
                    "scale_capped_error": 0.0,
                }
            )
            projected_update_loss = 0.0
            scale_capped_update_loss = 0.0
            for target_index, (target, owner, input_basis, group_bases) in enumerate(
                zip(
                    runtime.ranks.contract.targets,
                    runtime.owners,
                    input_bases,
                    output_bases,
                    strict=True,
                )
            ):
                a_name = target.name + LORA_A_SUFFIX
                b_name = target.name + LORA_B_SUFFIX
                reference_a = reference[a_name].float()
                reference_b = reference[b_name].float()
                projected_a = (reference_a @ input_basis) @ input_basis.transpose(0, 1)
                output_width = target.out_features // len(group_bases)
                projected_b_parts = []
                for group, basis in enumerate(group_bases):
                    block = reference_b[
                        group * output_width : (group + 1) * output_width
                    ]
                    projected_b_parts.append(basis @ (basis.transpose(0, 1) @ block))
                projected_b = torch.cat(projected_b_parts, dim=0)
                capped_a, capped_b, component_scale_ratios = _clip_singular_components(
                    a=projected_a,
                    b=projected_b,
                    scale_cap=runtime.ranks.s_ref[target_index],
                    in_features=target.in_features,
                    out_features=target.out_features,
                )
                reference_energy = _energy(reference_a, reference_b)
                projected_energy = _energy(projected_a, projected_b)
                projected_error = _distance_squared(
                    projected_a, projected_b, reference_a, reference_b
                )
                capped_error = _distance_squared(
                    capped_a, capped_b, reference_a, reference_b
                )
                normalizer = (
                    float(target.in_features * target.out_features)
                    * float(runtime.ranks.s_ref[target_index].square().detach())
                )
                weight = float(runtime.sensitivity_weights[member_index, target_index])
                projected_update_loss += weight * float(projected_error.detach()) / normalizer
                scale_capped_update_loss += weight * float(capped_error.detach()) / normalizer
                cell = family_totals[owner.family.value]
                cell["reference_energy"] += float(reference_energy.detach())
                cell["projected_energy"] += float(projected_energy.detach())
                cell["projected_error"] += float(projected_error.detach())
                cell["scale_capped_error"] += float(capped_error.detach())
                target_rows.append(
                    {
                        "target": target.name,
                        "family": owner.family.value,
                        "reference_energy": float(reference_energy.detach()),
                        "linear_span_energy_capture": float(
                            (projected_energy / reference_energy.clamp_min(1e-30)).detach()
                        ),
                        "linear_span_relative_error": math.sqrt(
                            float((projected_error / reference_energy.clamp_min(1e-30)).detach())
                        ),
                        "scale_capped_relative_error": math.sqrt(
                            float((capped_error / reference_energy.clamp_min(1e-30)).detach())
                        ),
                        "singular_component_to_s_ref_ratios": component_scale_ratios,
                    }
                )
                if member_name == args.materialize_member:
                    materialized_residual[a_name] = capped_a.detach().cpu()
                    materialized_residual[b_name] = capped_b.detach().cpu()
            family_rows = {}
            for family, values in sorted(family_totals.items()):
                reference_energy = max(values["reference_energy"], 1e-30)
                family_rows[family] = {
                    **values,
                    "linear_span_energy_capture": values["projected_energy"]
                    / reference_energy,
                    "linear_span_relative_error": math.sqrt(
                        max(values["projected_error"], 0.0) / reference_energy
                    ),
                    "scale_capped_relative_error": math.sqrt(
                        max(values["scale_capped_error"], 0.0) / reference_energy
                    ),
                }
            member_rows.append(
                {
                    "member": member_name,
                    "fixed50_successes": len(runtime.task.initial_success[member_name]),
                    "sensitivity_normalized_linear_span_update_loss": projected_update_loss,
                    "sensitivity_normalized_scale_capped_update_loss": scale_capped_update_loss,
                    "by_family": family_rows,
                    "targets": target_rows,
                }
            )

        materialized = None
        if args.materialize_member is not None:
            if len(materialized_residual) != 2 * len(runtime.ranks.contract.targets):
                raise ValueError("selected bank-span member was not materialized")
            complete = compose_rank12_plus_rank4(
                carrier_state={
                    name: value.detach().cpu()
                    for name, value in runtime.ranks.carrier_rank12.items()
                },
                residual_state=materialized_residual,
                rank16_contract=runtime.ranks.contract,
            )
            adapter = {
                name: value.to(device="cpu", dtype=torch.bfloat16).contiguous()
                for name, value in complete.items()
            }
            validate_lora_state(adapter, runtime.ranks.contract)
            materialized_root = args.output_dir / "materialized"
            materialized_root.mkdir()
            adapter_path = materialized_root / "adapter.safetensors"
            save_file(adapter, str(adapter_path))
            manifest_path = materialized_root / "manifest.json"
            write_json_atomic(
                manifest_path,
                {
                    "schema_version": G1_CHECKPOINT_SCHEMA,
                    "scientific_role": "read_only_g1_robust_bank_span_projection",
                    "step": 0,
                    "task_ordinal": runtime.task.ordinal,
                    "global_task_id": runtime.task.global_task_id,
                    "member": args.materialize_member,
                    "relative_singular_threshold": args.relative_singular_threshold,
                    "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
                    "single_complete_rank16": True,
                    "state_tensor_count": len(adapter),
                    "files": {"adapter.safetensors": adapter_path.stat().st_size},
                    "content_hash_policy": "disabled_by_owner",
                },
            )
            materialized = {
                "member": args.materialize_member,
                "adapter": str(adapter_path.resolve()),
                "adapter_bytes": adapter_path.stat().st_size,
                "manifest": str(manifest_path.resolve()),
                "manifest_bytes": manifest_path.stat().st_size,
                "single_complete_rank16": True,
            }

        return {
            "schema_version": SCHEMA,
            "scientific_role": "read_only_g1_native_bank_representability_diagnostic",
            "question": (
                "does the exact held-video native candidate bank contain the row and "
                "column spaces of known-success mobile-rank4 residuals before free-logit optimization"
            ),
            "claim_boundary": (
                "centered linear-span projection is the closure of two-softmax signed pooling; "
                "it does not prove that the current optimizer reaches the represented point"
            ),
            "repository": git_state(REPO_ROOT),
            "task": runtime.run_contract["task"],
            "video": runtime.run_contract["video"],
            "projection_relative_singular_threshold": args.relative_singular_threshold,
            "candidate_contract": {
                "input": ["frame", "probe", "horizon"],
                "output": ["frame", "probe", "horizon", "abs_adj_init_goal_type"],
                "q_output_groups": 8,
                "other_output_groups": 1,
                "centered_measure": True,
            },
            "target_spans": span_rows,
            "members": member_rows,
            "materialized_projection": materialized,
            "content_hash_policy": "disabled_by_owner",
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if runtime is not None:
            runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-ordinal", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-singular-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--materialize-member", choices=G1_MEMBER_NAMES, default=None
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("config", "asset_root", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not 0 < args.relative_singular_threshold < 1:
        raise ValueError("relative singular threshold must be in (0, 1)")
    args.mode = "profile"
    args.resume = None
    args.stop_after_step = 0
    result = analyze(args)
    write_json_atomic(args.output_dir / "analysis.json", result)
    write_json_atomic(
        args.output_dir / "completion.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "analysis": str((args.output_dir / "analysis.json").resolve()),
        },
    )
    print(
        json.dumps(
            {
                "task": result["task"]["ordinal"],
                "members": [
                    {
                        "member": row["member"],
                        "span_update_loss": row[
                            "sensitivity_normalized_linear_span_update_loss"
                        ],
                        "scale_capped_update_loss": row[
                            "sensitivity_normalized_scale_capped_update_loss"
                        ],
                    }
                    for row in result["members"]
                ],
                "output": str((args.output_dir / "analysis.json").resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
