"""GPU execution for the Program-Credit mechanism audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_stats
from ember.writer.adapter_analysis_metrics import (
    adapter_geometry,
    effective_metrics,
    effective_variance,
    lora_pairs,
    state_row,
    tensor_metrics,
)
from ember.writer.as_contract import REPO_ROOT
from ember.writer.coldstart_analysis import _fixed_action, _fixed_query
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.inference import (
    writer_generation_seed,
    writer_shuffled_frame_permutation,
)
from ember.writer.program_credit_analysis_metrics import (
    effective_difference_family,
    program_sketch,
    tensor_difference_family,
    tensor_variance,
)
from ember.writer.validation import _build_models


@dataclass
class AnalysisResources:
    policy: torch.nn.Module
    writer: torch.nn.Module
    lora: Any
    identity: Mapping[str, torch.Tensor]
    processor: Pi05LiberoProcessor
    tokenizer: Pi05TeacherPrefixTokenizer
    store: RawTeacherVideoStore
    tasks: dict[int, Mapping[str, Any]]
    by_id: dict[int, WriterTaskAuthority]
    wrong_by_id: dict[int, int]
    panel: set[int]
    pairs: Mapping[str, Mapping[str, str]]
    scale: float
    fixed_queries: dict[int, Mapping[str, torch.Tensor]]


def _condition_batch(
    task: WriterTaskAuthority,
    task_row: Mapping[str, Any],
    wrong_task_id: int,
    store: RawTeacherVideoStore,
    tokenizer: Pi05TeacherPrefixTokenizer,
    device: torch.device,
    generation_seed: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    frames: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    names: list[str] = []
    videos = {demo: store.load(task.task_id, demo) for demo in range(5)}

    def append(name: str, value: Any, order: torch.Tensor | None = None) -> None:
        selected = torch.from_numpy(value.frames)
        if order is not None:
            selected = selected.index_select(0, order)
        frames.append(selected)
        indices.append(torch.from_numpy(value.frame_indices))
        names.append(name)

    for demo in range(5):
        append(f"demo_{demo}", videos[demo])
    append("wrong_0", store.load(wrong_task_id, 0))
    count = int(videos[0].frames.shape[0])
    append("reversed_0", videos[0], torch.arange(count - 1, -1, -1))
    seed = writer_generation_seed(
        generation_seed,
        str(task_row["suite"]),
        int(task_row["task_id"]),
        0,
        stream="frame_order",
    )
    append(
        "shuffled_0",
        videos[0],
        writer_shuffled_frame_permutation(count, seed, keep_first=False),
    )
    offsets = [0]
    for value in frames:
        offsets.append(offsets[-1] + value.shape[0])
    tokens, masks, spans = tokenizer([task.language] * len(names))
    return (
        {
            "frames": torch.cat(frames).to(device, non_blocking=True),
            "indices": torch.cat(indices).to(device, non_blocking=True),
            "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
            "tokens": tokens,
            "masks": masks,
            "spans": spans,
        },
        names,
    )


def _build_resources(
    args: Any,
    context: Any,
    contract: Mapping[str, Any],
    as_training: Mapping[str, Any],
) -> AnalysisResources:
    tasks = {int(row["global_task_id"]): row for row in contract["tasks"]}
    authorities = tuple(
        WriterTaskAuthority(
            task_id=task_id,
            language=str(row["language"]),
            path=Path(row["path"]),
            expected_bytes=int(row["bytes"]),
            expected_sha256=None,
        )
        for task_id, row in tasks.items()
    )
    by_id = {task.task_id: task for task in authorities}
    policy, writer, lora, identity = _build_models(
        training=as_training,
        source=as_training["source"],
        context=context,
    )
    source_config = read_json(
        REPO_ROOT / str(as_training["authorities"]["source_base_config"]["path"])
    )
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    panel = set(int(value) for value in contract["action_panel_global_task_ids"])
    owned = [int(value) for value in contract["ownership"][context.rank]]
    return AnalysisResources(
        policy=policy,
        writer=writer,
        lora=lora,
        identity=identity,
        processor=processor,
        tokenizer=tokenizer,
        store=RawTeacherVideoStore(
            authorities,
            frame_stride=int(as_training["writer"]["frame_stride"]),
        ),
        tasks=tasks,
        by_id=by_id,
        wrong_by_id={
            int(row["language_global_task_id"]): int(row["video_global_task_id"])
            for row in contract["wrong_video_mapping"]
        },
        panel=panel,
        pairs=lora_pairs(writer),
        scale=float(lora.alpha) / float(lora.rank),
        fixed_queries={
            task_id: _fixed_query(by_id[task_id], processor, context.device)
            for task_id in owned
            if task_id in panel
        },
    )


def _checkpoint_outputs(
    resources: AnalysisResources,
    context: Any,
    task_id: int,
    packed: Mapping[str, torch.Tensor],
    names: list[str],
    record: Mapping[str, Any],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor] | None,
]:
    resources.writer.load_state_dict(
        load_file(
            str(Path(record["path"]) / "writer.safetensors"),
            device=str(context.device),
        ),
        strict=True,
    )
    resources.writer.eval()
    copy_task_lora_state_(resources.policy, resources.identity, resources.lora)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        program_batch = resources.writer.encode_program(
            packed["frames"],
            packed["indices"],
            packed["offsets"],
            packed["tokens"],
            packed["masks"],
            packed["spans"],
            policy=resources.policy,
        )
        state_batch = resources.writer.decode_program(program_batch)
    programs = {
        name: program_batch[index].detach().cpu().float()
        for index, name in enumerate(names)
    }
    states = {
        name: state_row(state_batch, index) for index, name in enumerate(names)
    }
    actions = None
    if task_id in resources.panel:
        action_names = ("demo_0", "demo_1", "wrong_0", "reversed_0", "shuffled_0")
        actions = {
            name: _fixed_action(
                resources.policy,
                resources.processor,
                resources.fixed_queries[task_id],
                states[name],
                resources.lora,
                202608050000 + task_id,
                context.device,
            )
            for name in action_names
        }
    return programs, states, actions


def _base_row(
    resources: AnalysisResources,
    context: Any,
    task_id: int,
    record: Mapping[str, Any],
    programs: Mapping[str, torch.Tensor],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    actions: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    task_row = resources.tasks[task_id]
    video_names = tuple(f"demo_{demo}" for demo in range(5))
    action_names = ("demo_1", "wrong_0", "reversed_0", "shuffled_0")
    return {
        "checkpoint_cursor": int(record["checkpoint_cursor"]),
        "checkpoint_label": str(record["label"]),
        "source_cursor": int(record["source_cursor"]),
        "global_task_id": task_id,
        "suite": str(task_row["suite"]),
        "suite_task_id": int(task_row["task_id"]),
        "rank": context.rank,
        "geometry_demo_0": adapter_geometry(
            resources.writer, resources.pairs, states["demo_0"], resources.scale
        ),
        "program_same_task_video_variance": tensor_variance(
            [programs[name] for name in video_names]
        ),
        "effective_ba_same_task_video_variance": effective_variance(
            resources.pairs, [states[name] for name in video_names]
        ),
        "program_condition_from_demo_0": {
            name: tensor_metrics(programs["demo_0"], value)
            for name, value in programs.items()
            if name != "demo_0"
        },
        "effective_ba_condition_from_demo_0": {
            name: effective_metrics(resources.pairs, states["demo_0"], value)
            for name, value in states.items()
            if name != "demo_0"
        },
        "fixed_action_condition_from_demo_0": (
            {
                name: tensor_metrics(actions["demo_0"], actions[name])
                for name in action_names
            }
            if actions is not None
            else None
        ),
        "checkpoint_update": None,
        "program_credit": None,
    }


def _transition(
    resources: AnalysisResources,
    task_id: int,
    programs: Mapping[str, torch.Tensor],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    actions: Mapping[str, torch.Tensor] | None,
    baseline_programs: Mapping[int, Mapping[str, torch.Tensor]],
    baseline_states: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    baseline_actions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    base_program = baseline_programs[task_id]
    base_state = baseline_states[task_id]
    video_names = tuple(f"demo_{demo}" for demo in range(5))
    program_family, program_mean_delta = tensor_difference_family(
        base_program, programs, video_names
    )
    return {
        "program_demo_0": tensor_metrics(base_program["demo_0"], programs["demo_0"]),
        "effective_ba_demo_0": effective_metrics(
            resources.pairs, base_state["demo_0"], states["demo_0"]
        ),
        "program_by_condition": {
            name: tensor_metrics(base_program[name], programs[name]) for name in programs
        },
        "effective_ba_by_condition": {
            name: effective_metrics(resources.pairs, base_state[name], states[name])
            for name in states
        },
        "program_same_task_update_family": program_family,
        "effective_ba_same_task_update_family": effective_difference_family(
            resources.pairs, base_state, states, video_names
        ),
        "program_task_mean_delta_sketch": program_sketch(program_mean_delta),
        "fixed_action_demo_0": (
            tensor_metrics(baseline_actions[task_id]["demo_0"], actions["demo_0"])
            if actions is not None
            else None
        ),
    }


def _credit_record(reward_run: Path, global_task_id: int) -> dict[str, Any]:
    row = read_json(
        reward_run
        / "program_credit/cycle_00000000"
        / f"task_{global_task_id:03d}.json"
    )
    return {
        "direction_seeds": [int(pair["direction_seed"]) for pair in row["pairs"]],
        "credits": [float(pair["credit"]) for pair in row["pairs"]],
        "credit_modes": [str(pair["credit_mode"]) for pair in row["pairs"]],
        "cotangent_norm": float(row["cotangent_norm"]),
    }


def local_rows(
    args: Any,
    context: Any,
    contract: Mapping[str, Any],
    as_training: Mapping[str, Any],
) -> list[dict[str, Any]]:
    resources = _build_resources(args, context, contract, as_training)
    owned = [int(value) for value in contract["ownership"][context.rank]]
    baseline_programs: dict[int, Mapping[str, torch.Tensor]] = {}
    baseline_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    baseline_actions: dict[int, Mapping[str, torch.Tensor]] = {}
    rows = []
    try:
        for task_id in owned:
            packed, names = _condition_batch(
                resources.by_id[task_id],
                resources.tasks[task_id],
                resources.wrong_by_id[task_id],
                resources.store,
                resources.tokenizer,
                context.device,
                int(as_training["writer"]["initialization_seed"]),
            )
            for record in contract["checkpoints"]:
                programs, states, actions = _checkpoint_outputs(
                    resources, context, task_id, packed, names, record
                )
                row = _base_row(
                    resources, context, task_id, record, programs, states, actions
                )
                if record["label"] == "as125":
                    baseline_programs[task_id] = programs
                    baseline_states[task_id] = states
                    if actions is not None:
                        baseline_actions[task_id] = actions
                else:
                    row["checkpoint_update"] = _transition(
                        resources,
                        task_id,
                        programs,
                        states,
                        actions,
                        baseline_programs,
                        baseline_states,
                        baseline_actions,
                    )
                    row["program_credit"] = _credit_record(
                        args.reward_training_run, task_id
                    )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "rank": context.rank,
                            "checkpoint": record["label"],
                            "task": task_id,
                        }
                    ),
                    flush=True,
                )
    finally:
        copy_task_lora_state_(resources.policy, resources.identity, resources.lora)
        resources.store.close()
    return rows
