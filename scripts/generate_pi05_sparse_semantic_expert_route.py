#!/usr/bin/env python3
"""Generate the frozen train24 semantic route for sparse Trace-M2P experts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.writer.video_program import Pi05FrozenConditionDescriptor


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spherical_kmeans(
    anchors: torch.Tensor,
    *,
    expert_count: int,
    seed: int,
    maximum_iterations: int,
) -> tuple[torch.Tensor, list[int], int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.randint(anchors.shape[0], (1,), generator=generator))]
    while len(chosen) < expert_count:
        similarity = anchors @ anchors[chosen].T
        nearest = similarity.max(dim=1).values
        nearest[chosen] = torch.inf
        chosen.append(int(nearest.argmin()))
    centers = anchors[chosen].clone()
    previous: torch.Tensor | None = None
    for iteration in range(1, maximum_iterations + 1):
        assignment = (anchors @ centers.T).argmax(dim=1)
        if previous is not None and torch.equal(assignment, previous):
            return centers, chosen, iteration
        previous = assignment
        centers = torch.stack(
            [
                F.normalize(anchors[assignment == index].mean(dim=0), dim=0)
                for index in range(expert_count)
            ]
        )
    raise RuntimeError("sparse semantic expert route did not converge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/"
            "checkpoints/step_00001000"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "models/tokenizers/openpi/paligemma_tokenizer.model",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=REPO_ROOT / "configs/pi05_target_data_v1/manifest.json",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_source_base_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs/pi05_sparse_semantic_expert_route_v1.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    source_config = read_json(args.source_config)
    policy = load_policy(args.checkpoint / "policy", source_config, device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    rows = sorted(
        (
            row
            for row in read_json(args.target_manifest)["tasks"]
            if row["split_role"] == "train"
        ),
        key=lambda row: int(row["global_task_id"]),
    )
    if len(rows) != 24:
        raise RuntimeError("semantic route requires the sealed train24 split")
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    tokens, mask, task_span = tokenizer([str(row["language"]) for row in rows])
    descriptor = Pi05FrozenConditionDescriptor(
        image_width=2048,
        expert_width=1024,
        max_frames_per_encoder_call=16,
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=7,
    ).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        anchors = descriptor.task_anchor(policy, tokens, mask, task_span)
        singleton_anchors = []
        for row in rows:
            single_tokens, single_mask, single_task_span = tokenizer(
                [str(row["language"])]
            )
            singleton_anchors.append(
                descriptor.task_anchor(
                    policy,
                    single_tokens,
                    single_mask,
                    single_task_span,
                )
            )
        singleton_anchors = torch.cat(singleton_anchors)
    cobatch_max_abs = float((anchors - singleton_anchors).abs().max())
    if cobatch_max_abs > 1e-3:
        raise RuntimeError("semantic anchor materially changed with language co-batching")
    batched_anchors = anchors.float().cpu()
    anchors = singleton_anchors.float().cpu()
    anchor_mean = anchors.mean(dim=0)
    centered = F.normalize(anchors - anchor_mean, dim=-1)
    centers, initial_indices, iterations = _spherical_kmeans(
        centered,
        expert_count=8,
        seed=7,
        maximum_iterations=100,
    )
    similarities = centered @ centers.T
    values, indices = torch.topk(similarities, k=2, dim=1, sorted=True)
    batched_indices = torch.topk(
        F.normalize(batched_anchors - anchor_mean, dim=-1) @ centers.T,
        k=2,
        dim=1,
        sorted=True,
    ).indices
    if not torch.equal(indices, batched_indices):
        raise RuntimeError("semantic route changed with language co-batching")
    primary_counts = torch.bincount(indices[:, 0], minlength=8)
    top2_counts = torch.bincount(indices.flatten(), minlength=8)
    if int(primary_counts.min()) <= 0 or int(top2_counts.min()) <= 0:
        raise RuntimeError("semantic route expert usage collapsed")
    task_routes = [
        {
            "global_task_id": int(row["global_task_id"]),
            "language": str(row["language"]),
            "primary_expert": int(indices[index, 0]),
            "secondary_expert": int(indices[index, 1]),
            "primary_cosine": float(values[index, 0]),
            "secondary_cosine": float(values[index, 1]),
        }
        for index, row in enumerate(rows)
    ]
    write_json_atomic(
        args.output,
        {
            "schema_version": "ember_pi05_sparse_semantic_expert_route_v1",
            "fit": {
                "task_roles": ["train"],
                "task_count": 24,
                "seed": 7,
                "method": "spherical_kmeans",
                "expert_count": 8,
                "top_k": 2,
                "anchor": (
                    "train24_mean_centered_frozen_base_text_task_span_mean_l2"
                ),
            },
            "algorithm": {
                "initialization": (
                    "seed7_first_anchor_then_deterministic_farthest_cosine"
                ),
                "maximum_iterations": 100,
                "converged_iterations": iterations,
                "assignment": "maximum_cosine_argmax_lowest_index_tie",
                "center_update": "l2_normalized_member_mean",
            },
            "audit": {
                "initial_task_indices": initial_indices,
                "primary_expert_counts": primary_counts.tolist(),
                "top2_expert_counts": top2_counts.tolist(),
                "top2_usage_limit": 12,
                "exact_language_cobatch_invariant": True,
                "anchor_cobatch_max_abs": cobatch_max_abs,
            },
            "anchor_mean": anchor_mean.tolist(),
            "centers": centers.tolist(),
            "task_routes": task_routes,
        },
    )


if __name__ == "__main__":
    main()
