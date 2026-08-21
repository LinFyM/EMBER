"""Fixed correct-only observer panel for a frozen ECP Stage 0 checkpoint."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ember.ecp.checkpoint import checkpoint_macro
from ember.ecp.contracts import build_target_owners
from ember.ecp.stage0 import ECPStage0Model
from ember.ecp.stage0_training import load_stage0_config
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_SCHEMA = "ember_ecp_stage0_observer_panel_v1"


@dataclass(frozen=True)
class ObserverPanelTask:
    task_id: int
    role: str
    suite: str
    language: str
    authority: WriterTaskAuthority


def _panel_tasks(
    config: dict[str, Any], data_root: Path
) -> tuple[ObserverPanelTask, ...]:
    manifest = read_json(REPO_ROOT / config["authorities"]["target_manifest"])
    fit = set(map(int, config["task_roles"]["target_fit_task_ids"]))
    held = set(map(int, config["task_roles"]["target_held_task_ids"]))
    requested = fit | held
    rows = []
    for record in manifest["tasks"]:
        task_id = int(record["global_task_id"])
        if task_id not in requested:
            continue
        hdf5 = record["hdf5"]
        rows.append(
            ObserverPanelTask(
                task_id=task_id,
                role="fit19" if task_id in fit else "held5",
                suite=str(record["suite"]),
                language=str(record["language"]),
                authority=WriterTaskAuthority(
                    task_id=task_id,
                    language=str(record["language"]),
                    path=data_root / str(hdf5["relative_path"]),
                    expected_bytes=int(hdf5["bytes"]),
                ),
            )
        )
    rows.sort(key=lambda task: task.task_id)
    if len(rows) != 24 or {task.task_id for task in rows} != requested:
        raise ValueError("ECP observer panel is not the fixed train24 fold")
    return tuple(rows)


def _ordered_speed_view(frames: Any, factor: int) -> Any:
    selected = list(range(0, int(frames.shape[0]), factor))
    if selected[-1] != frames.shape[0] - 1:
        selected.append(int(frames.shape[0] - 1))
    return frames[selected]


def _pack_views(
    store: RawTeacherVideoStore,
    *,
    task_id: int,
    primary_demo: int,
    other_demo: int,
    speed_factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    primary = store.load(task_id, primary_demo).frames
    other = store.load(task_id, other_demo).frames
    views = (primary, _ordered_speed_view(primary, speed_factor), other)
    counts = [int(value.shape[0]) for value in views]
    frames = torch.from_numpy(np.concatenate(views)).to(
        device=device, non_blocking=True
    )
    offsets = torch.tensor(
        [0, counts[0], counts[0] + counts[1], sum(counts)],
        dtype=torch.long,
        device=device,
    )
    return frames, offsets, counts


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)
    )


def _event_cosine(
    left: torch.Tensor,
    left_presence: torch.Tensor,
    right: torch.Tensor,
    right_presence: torch.Tensor,
) -> float:
    values = F.cosine_similarity(
        left.float().flatten(1), right.float().flatten(1), dim=1
    )
    weights = (left_presence.float() * right_presence.float()).clamp_min(0).sqrt()
    weights = weights / weights.sum().clamp_min(1e-6)
    return float((values * weights).sum())


def _mean(rows: Iterable[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return sum(values) / len(values)


def _summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = tuple(
        name
        for name in rows[0]
        if name.endswith("_cosine")
        or name.endswith("_margin")
        or name.endswith("_l1")
    )
    result = {}
    for role in ("all", "fit19", "held5"):
        selected = rows if role == "all" else [row for row in rows if row["role"] == role]
        result[role] = {
            "rows": len(selected),
            **{name: _mean(selected, name) for name in metric_names},
        }
    return result


def _load_model(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    source_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ECPStage0Model]:
    source_config = load_config(REPO_ROOT / config["authorities"]["source_base_config"])
    policy = load_policy(source_checkpoint / "policy", source_config, device)
    policy.requires_grad_(False).eval()
    owners = build_target_owners(
        load_pi05_lora_contract(REPO_ROOT / config["authorities"]["lora_contract"])
    )
    cell = config["model"]
    model = ECPStage0Model(
        owners,
        prefix_width=int(cell["prefix_width"]),
        expert_width=int(cell["expert_width"]),
        program_width=int(cell["program_width"]),
        event_slots=int(cell["event_slots"]),
        action_phases=int(cell["action_phases"]),
        max_frames_per_call=int(cell["max_frames_per_call"]),
        fixed_probe_seed=int(cell["fixed_probe_seed"]),
    ).to(device)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    weights = checkpoint / "ecp.safetensors"
    if (
        manifest.get("stage") != "stage0_native"
        or int(manifest.get("next_macro", -1)) != checkpoint_macro(checkpoint)
        or not weights.is_file()
        or weights.stat().st_size != int(manifest["files"][weights.name]["bytes"])
    ):
        raise ValueError("ECP observer checkpoint authority changed")
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    return policy, model.eval()


def _evaluate_pair(
    *,
    model: ECPStage0Model,
    policy: torch.nn.Module,
    store: RawTeacherVideoStore,
    task: ObserverPanelTask,
    pair_ordinal: int,
    primary_demo: int,
    other_demo: int,
    speed_factor: int,
    language_tokens: torch.Tensor,
    language_mask: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    frames, offsets, counts = _pack_views(
        store,
        task_id=task.task_id,
        primary_demo=primary_demo,
        other_demo=other_demo,
        speed_factor=speed_factor,
        device=device,
    )
    frame_condition_ids = torch.zeros(
        frames.shape[0], dtype=torch.long, device=device
    )
    common = {
        "policy": policy,
        "language_tokens": language_tokens,
        "language_mask": language_mask,
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        process, presence, _, _, _, _, _, summary = model.encoder(
            frames=frames,
            video_offsets=offsets,
            frame_condition_ids=frame_condition_ids,
            **common,
        )
        anti_process, anti_presence, _, _, _, _, _, anti_summary = model.encoder(
            frames=frames[: counts[0]],
            video_offsets=torch.tensor(
                [0, counts[0]], dtype=torch.long, device=device
            ),
            frame_condition_ids=frame_condition_ids[: counts[0]],
            suffix_noise=-model.encoder.fixed_suffix_noise,
            **common,
        )
    row = {
        "task_id": task.task_id,
        "role": task.role,
        "suite": task.suite,
        "pair_ordinal": pair_ordinal,
        "primary_demo": primary_demo,
        "other_demo": other_demo,
        "sampled_frames": counts,
        "speed_summary_cosine": _cosine(summary[0], summary[1]),
        "speed_event_cosine": _event_cosine(
            process[0], presence[0], process[1], presence[1]
        ),
        "other_summary_cosine": _cosine(summary[0], summary[2]),
        "other_event_cosine": _event_cosine(
            process[0], presence[0], process[2], presence[2]
        ),
        "antithetic_summary_cosine": _cosine(summary[0], anti_summary[0]),
        "antithetic_event_cosine": _event_cosine(
            process[0], presence[0], anti_process[0], anti_presence[0]
        ),
        "speed_presence_l1": float(
            (presence[0].float() - presence[1].float()).abs().mean()
        ),
        "other_presence_l1": float(
            (presence[0].float() - presence[2].float()).abs().mean()
        ),
        "antithetic_presence_l1": float(
            (presence[0].float() - anti_presence[0].float()).abs().mean()
        ),
    }
    return row, summary[0].float().cpu()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ECP observer panel requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = load_stage0_config(args.config)
    panel = config["observer_panel"]
    tasks = _panel_tasks(config, args.data_root)
    policy, model = _load_model(
        config=config,
        checkpoint=args.checkpoint,
        source_checkpoint=args.source_checkpoint,
        device=device,
    )
    source_config = load_config(REPO_ROOT / config["authorities"]["source_base_config"])
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    tokens, masks, _ = tokenizer([task.language for task in tasks])
    store = RawTeacherVideoStore(
        [task.authority for task in tasks],
        frame_stride=int(config["data"]["frame_stride"]),
    )
    rows: list[dict[str, Any]] = []
    canonical_summaries: dict[tuple[int, int], torch.Tensor] = {}
    started = time.monotonic()
    try:
        for task_index, task in enumerate(tasks):
            language_tokens = tokens[task_index : task_index + 1]
            language_mask = masks[task_index : task_index + 1]
            for pair_ordinal, (primary_demo, other_demo) in enumerate(
                panel["demo_pairs"]
            ):
                row, canonical = _evaluate_pair(
                    model=model,
                    policy=policy,
                    store=store,
                    task=task,
                    pair_ordinal=pair_ordinal,
                    primary_demo=int(primary_demo),
                    other_demo=int(other_demo),
                    speed_factor=int(panel["ordered_speed_factor"]),
                    language_tokens=language_tokens,
                    language_mask=language_mask,
                    device=device,
                )
                canonical_summaries[(task.task_id, pair_ordinal)] = canonical
                rows.append(row)
    finally:
        store.close()
    for row in rows:
        key = (int(row["task_id"]), int(row["pair_ordinal"]))
        canonical = canonical_summaries[key]
        cross = [
            _cosine(canonical, value)
            for (task_id, ordinal), value in canonical_summaries.items()
            if ordinal == key[1] and task_id != key[0]
        ]
        row["mean_cross_task_summary_cosine"] = sum(cross) / len(cross)
        row["nearest_cross_task_summary_cosine"] = max(cross)
        row["mean_cross_task_margin"] = (
            float(row["other_summary_cosine"])
            - row["mean_cross_task_summary_cosine"]
        )
        row["nearest_cross_task_margin"] = (
            float(row["other_summary_cosine"])
            - row["nearest_cross_task_summary_cosine"]
        )
    result = {
        "schema_version": PANEL_SCHEMA,
        "git": git_state(REPO_ROOT),
        "checkpoint": str(args.checkpoint),
        "checkpoint_macro": checkpoint_macro(args.checkpoint),
        "source_checkpoint": str(args.source_checkpoint),
        "conditions": [
            "canonical correct video",
            "same video order-preserving 2x speed",
            "same-task other correct video",
            "same video antithetic fixed Gaussian probe",
        ],
        "forbidden_conditions_used": [],
        "task_count": len(tasks),
        "row_count": len(rows),
        "rows": rows,
        "aggregates": _summaries(rows),
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    if args.output.exists():
        raise ValueError("ECP observer panel output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_stage0_native_v1.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "config",
        "checkpoint",
        "source_checkpoint",
        "tokenizer_path",
        "data_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = evaluate(args)
    print({"output": str(args.output), "aggregates": result["aggregates"]})


if __name__ == "__main__":
    main()
