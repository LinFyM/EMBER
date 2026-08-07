#!/usr/bin/env python3
"""Extract train24 grounded-video addresses and seal a fixed top2 route."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.video_program import Pi05FrozenConditionDescriptor


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--aggregate", action="store_true")
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
        "--data-root",
        type=Path,
        default=(
            REPO_ROOT
            / "data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs/pi05_grounded_video_expert_route_v1.json",
    )
    return parser.parse_args()


def _train_rows(path: Path) -> list[dict]:
    rows = sorted(
        (
            row
            for row in read_json(path)["tasks"]
            if row["split_role"] == "train"
        ),
        key=lambda row: int(row["global_task_id"]),
    )
    if len(rows) != 24:
        raise RuntimeError("grounded-video route requires sealed train24")
    return rows


@torch.inference_mode()
def _video_addresses(
    descriptor: Pi05FrozenConditionDescriptor,
    policy: torch.nn.Module,
    tokenizer: Pi05TeacherPrefixTokenizer,
    videos: list,
    language: str,
    device: torch.device,
) -> torch.Tensor:
    frames = [torch.from_numpy(video.frames).to(device, non_blocking=True) for video in videos]
    offsets = [0]
    for value in frames:
        offsets.append(offsets[-1] + int(value.shape[0]))
    video_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    lengths = video_offsets[1:] - video_offsets[:-1]
    frame_video_ids = torch.repeat_interleave(
        torch.arange(len(videos), device=device), lengths
    )
    tokens, masks, spans = tokenizer([language] * len(videos))
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        _, addresses = descriptor(
            policy,
            torch.cat(frames),
            frame_video_ids,
            video_offsets,
            tokens,
            masks,
            spans,
        )
    return addresses.float().cpu()


def _extract(args: argparse.Namespace) -> None:
    if args.rank is None or args.world_size is None or not (
        0 <= args.rank < args.world_size <= 6
    ):
        raise RuntimeError("route extraction requires explicit rank/world-size")
    device = torch.device("cuda:0")
    source = read_json(args.source_config)
    policy = load_policy(args.checkpoint / "policy", source, device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source["features"]["tokenizer_max_length"]),
        str(device),
    )
    descriptor = Pi05FrozenConditionDescriptor(
        image_width=2048,
        expert_width=1024,
        max_frames_per_encoder_call=16,
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=7,
    ).to(device)
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    rows = _train_rows(args.target_manifest)
    for ordinal, row in enumerate(rows):
        if ordinal % args.world_size != args.rank:
            continue
        task_id = int(row["global_task_id"])
        authority = WriterTaskAuthority(
            task_id=task_id,
            language=str(row["language"]),
            path=args.data_root / str(row["hdf5"]["relative_path"]),
            expected_bytes=int(row["hdf5"]["bytes"]),
        )
        store = RawTeacherVideoStore((authority,), frame_stride=5, max_open_files=1)
        addresses = []
        first_videos = []
        for start in range(0, 50, 4):
            videos = [store.load(task_id, index) for index in range(start, min(start + 4, 50))]
            if start == 0:
                first_videos = videos
            addresses.append(
                _video_addresses(
                    descriptor,
                    policy,
                    tokenizer,
                    videos,
                    authority.language,
                    device,
                )
            )
        singleton = torch.cat(
            [
                _video_addresses(
                    descriptor,
                    policy,
                    tokenizer,
                    [video],
                    authority.language,
                    device,
                )
                for video in first_videos
            ]
        )
        value = torch.cat(addresses)
        if value.shape != (50, 2048) or singleton.shape != (4, 2048):
            raise RuntimeError("grounded-video address extraction changed shape")
        save_file(
            {
                "video_addresses": value.to(torch.bfloat16),
                "singleton_audit_addresses": singleton.to(torch.bfloat16),
                "demo_indices": torch.arange(50, dtype=torch.int64),
            },
            str(args.extract_dir / f"task_{task_id:03d}.safetensors"),
        )
        store.close()
        print(f"rank={args.rank} task={task_id} complete", flush=True)


def _spherical_kmeans(
    anchors: torch.Tensor,
    *,
    expert_count: int,
    seed: int,
) -> tuple[torch.Tensor, list[int], int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.randint(anchors.shape[0], (1,), generator=generator))]
    while len(chosen) < expert_count:
        nearest = (anchors @ anchors[chosen].T).max(dim=1).values
        nearest[chosen] = torch.inf
        chosen.append(int(nearest.argmin()))
    centers = anchors[chosen].clone()
    previous = None
    for iteration in range(1, 101):
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
    raise RuntimeError("grounded-video route did not converge")


def _aggregate(args: argparse.Namespace) -> None:
    rows = _train_rows(args.target_manifest)
    addresses = []
    singleton = []
    for row in rows:
        task_id = int(row["global_task_id"])
        path = args.extract_dir / f"task_{task_id:03d}.safetensors"
        value = load_file(str(path))
        address = F.normalize(value["video_addresses"].float(), dim=-1)
        audit = F.normalize(value["singleton_audit_addresses"].float(), dim=-1)
        if address.shape != (50, 2048) or audit.shape != (4, 2048):
            raise RuntimeError("grounded-video route extraction is incomplete")
        addresses.append(address)
        singleton.append(audit)
    prototypes = F.normalize(
        torch.stack([value.mean(dim=0) for value in addresses]), dim=-1
    )
    anchor_mean = prototypes.mean(dim=0)
    centered = F.normalize(prototypes - anchor_mean, dim=-1)
    centers, initial_indices, iterations = _spherical_kmeans(
        centered, expert_count=8, seed=7
    )
    similarities = centered @ centers.T
    values, indices = torch.topk(similarities, k=2, dim=1, sorted=True)
    primary_counts = torch.bincount(indices[:, 0], minlength=8)
    top2_counts = torch.bincount(indices.flatten(), minlength=8)

    primary_matches = []
    exact_matches = []
    overlaps = []
    for task_ordinal, (row, task_addresses) in enumerate(zip(rows, addresses, strict=True)):
        reference = indices[task_ordinal]
        task_id = int(row["global_task_id"])
        for sample in range(250):
            selected = torch.randperm(
                50,
                generator=torch.Generator(device="cpu").manual_seed(
                    100_000 * task_id + sample
                ),
            )[:4]
            query = F.normalize(task_addresses[selected].mean(dim=0), dim=0)
            route = torch.topk(
                F.normalize(query - anchor_mean, dim=0) @ centers.T,
                k=2,
            ).indices
            primary_matches.append(float(route[0] == reference[0]))
            exact_matches.append(float(torch.equal(route, reference)))
            overlaps.append(
                len(set(route.tolist()) & set(reference.tolist())) / 2.0
            )
    primary_stability = sum(primary_matches) / len(primary_matches)
    exact_stability = sum(exact_matches) / len(exact_matches)
    top2_overlap = sum(overlaps) / len(overlaps)

    cobatch_routes = []
    for task_ordinal, (batch, single) in enumerate(
        zip(addresses, singleton, strict=True)
    ):
        batch_query = F.normalize(batch[:4].mean(dim=0), dim=0)
        single_query = F.normalize(single.mean(dim=0), dim=0)
        batch_route = torch.topk(
            F.normalize(batch_query - anchor_mean, dim=0) @ centers.T, k=2
        ).indices
        single_route = torch.topk(
            F.normalize(single_query - anchor_mean, dim=0) @ centers.T, k=2
        ).indices
        cobatch_routes.append(
            {
                "global_task_id": int(rows[task_ordinal]["global_task_id"]),
                "batch4": batch_route.tolist(),
                "singleton": single_route.tolist(),
                "exact": bool(torch.equal(batch_route, single_route)),
                "address_max_abs": float((batch_query - single_query).abs().max()),
            }
        )
    if (
        int(primary_counts.min()) <= 0
        or int(top2_counts.min()) <= 0
        or primary_stability < 0.90
        or top2_overlap < 0.90
        or not all(row["exact"] for row in cobatch_routes)
    ):
        raise RuntimeError("grounded-video route failed its input-only gate")

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
            "schema_version": "ember_pi05_grounded_video_expert_route_v1",
            "fit": {
                "task_roles": ["train"],
                "task_count": 24,
                "seed": 7,
                "method": "spherical_kmeans",
                "expert_count": 8,
                "top_k": 2,
                "anchor": "train24_mean_centered_k4_multimodal_task_token_video_innovation_l2",
            },
            "algorithm": {
                "initialization": "seed7_first_anchor_then_deterministic_farthest_cosine",
                "maximum_iterations": 100,
                "converged_iterations": iterations,
                "assignment": "maximum_cosine_argmax_lowest_index_tie",
                "center_update": "l2_normalized_member_mean",
                "runtime_video_pool": "per_video_frame_mean_l2_then_k4_mean_l2",
            },
            "audit": {
                "initial_task_indices": initial_indices,
                "primary_expert_counts": primary_counts.tolist(),
                "top2_expert_counts": top2_counts.tolist(),
                "random_k4_sets_per_task": 250,
                "primary_route_stability": primary_stability,
                "exact_top2_route_stability": exact_stability,
                "mean_top2_overlap": top2_overlap,
                "minimum_primary_stability": 0.90,
                "minimum_top2_overlap": 0.90,
                "batch4_singleton_routes_exact": True,
                "cobatch_routes": cobatch_routes,
                "teacher_action_state_reward_terminal_reads": 0,
                "validation_test_video_reads": 0,
            },
            "anchor_mean": anchor_mean.tolist(),
            "centers": centers.tolist(),
            "task_routes": task_routes,
        },
    )
    print(
        f"primary={primary_stability:.6f} exact={exact_stability:.6f} "
        f"overlap={top2_overlap:.6f} primary_counts={primary_counts.tolist()} "
        f"top2_counts={top2_counts.tolist()}",
        flush=True,
    )


def main() -> None:
    args = _args()
    if args.aggregate:
        _aggregate(args)
    else:
        _extract(args)


if __name__ == "__main__":
    main()
