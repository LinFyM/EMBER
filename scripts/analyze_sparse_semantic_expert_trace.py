#!/usr/bin/env python3
"""Probe the sealed sparse-expert Writer from K4 traces to fixed policy action."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_
from ember.pi05_eval.worker_setup import load_policy
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.adapter_analysis_metrics import (
    adapter_geometry,
    distribution,
    effective_metrics,
    lora_pairs,
    tensor_metrics,
)
from ember.writer.data import _camera
from ember.writer.fewshot_m2p import factorize_trace_evidence
from ember.writer.live_adapter import FrozenWriterTaskAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("correct", "same", "wrong", "shuffled", "reversed")
ROOTS = {
    "correct": "pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_bci_correct400_noreplacement_seed7_macro0150_3820f27_20260807",
    "same": "pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_bci_same_task_other400_noreplacement_seed7_macro0150_3820f27_20260807",
    "wrong": "pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_bci_cross_suite_wrong400_noreplacement_seed7_macro0150_3820f27_20260807",
    "shuffled": "pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_bci_shuffled400_noreplacement_seed7_macro0150_3820f27_20260807",
    "reversed": "pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_bci_reversed400_noreplacement_seed7_macro0150_3820f27_20260807",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index", type=int)
    group.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _contracts() -> dict[str, dict[str, Any]]:
    return {
        condition: read_json(REPO_ROOT / "runs/outputs" / root / "run_contract.json")
        for condition, root in ROOTS.items()
    }


def _fixed_query(
    contract: Mapping[str, Any],
    processor: Any,
    *,
    suite: str,
    task_id: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    mapping = next(
        row
        for row in contract["adapter"]["task_video_mapping"]
        if row["suite"] == suite and int(row["task_id"]) == task_id
    )
    global_task_id = int(mapping["language_global_task_id"])
    manifest = read_json(REPO_ROOT / "configs/pi05_target_data_v1/manifest.json")
    task = next(
        row for row in manifest["tasks"]
        if int(row["global_task_id"]) == global_task_id
    )
    path = Path(contract["adapter"]["video_data"]["root"]) / task["hdf5"]["relative_path"]
    with h5py.File(path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        base = np.asarray(obs["agentview_rgb"][0])
        wrist = np.asarray(obs["eye_in_hand_rgb"][0])
        state = np.concatenate(
            (
                np.asarray(obs["ee_states"][0], dtype=np.float32),
                np.asarray(obs["gripper_states"][0], dtype=np.float32),
            )
        )
    states = torch.from_numpy(state)[None].to(device)
    tokens, masks = processor._tokenize_prompts(states, [str(task["language"])])
    return {
        "observation.images.base_0_rgb": torch.from_numpy(_camera(base))[None]
        .to(device, dtype=torch.float32)
        .div_(255.0),
        "observation.images.left_wrist_0_rgb": torch.from_numpy(_camera(wrist))[None]
        .to(device, dtype=torch.float32)
        .div_(255.0),
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": masks,
    }, global_task_id


@torch.inference_mode()
def _policy_action(
    adapter: FrozenWriterTaskAdapter,
    processor: Any,
    prepared: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    policy = adapter.policy
    bridge = policy.model.paligemma_with_expert
    configs = (
        bridge.paligemma.model.language_model.config,
        bridge.gemma_expert.model.config,
    )
    backends = tuple(str(value._attn_implementation) for value in configs)
    copy_task_lora_state_(policy, state, adapter.lora_contract)
    noise = torch.randn(
        1,
        int(policy.model.config.chunk_size),
        int(policy.model.config.max_action_dim),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(adapter.device)
    try:
        with torch.autocast(
            device_type=adapter.device.type,
            dtype=torch.bfloat16,
            enabled=adapter.device.type == "cuda",
        ):
            action = policy.predict_action_chunk(
                dict(prepared), noise=noise, num_steps=10
            )
    finally:
        for config, backend in zip(configs, backends, strict=True):
            config._attn_implementation = backend
        copy_task_lora_state_(
            policy, adapter.identity_state, adapter.lora_contract
        )
    return processor.unnormalize_action(action).detach().float().cpu()


def _capture_task(
    task_index: int,
    output_dir: Path,
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    correct = contracts["correct"]
    tasks = tuple(
        (str(row["suite"]), int(row["task_id"])) for row in correct["tasks"]
    )
    if not 0 <= task_index < len(tasks):
        raise ValueError("task index is outside the validation panel")
    suite, task_id = tasks[task_index]
    device = torch.device("cuda:0")
    normalization = read_json(Path(correct["normalization"]["path"]))["stats"]
    policy, processor, _ = load_policy(
        Path(correct["model"]["model_path"]),
        normalization,
        Path(correct["tokenizer"]["path"]),
        correct["policy"],
    )
    adapter = FrozenWriterTaskAdapter(
        policy=policy,
        source=correct["model"],
        evaluation_adapter=correct["adapter"],
        task_keys=tasks,
        device=device,
        tokenizer_path=Path(correct["tokenizer"]["path"]),
        require_formal=True,
    )
    evidence = {}
    physical_rows = []
    direction_rows = []
    trace_evidence_rows = []
    readers = []
    programs = []
    states = []
    routes = []
    route_weights_rows = []
    writer = adapter.writer
    local_offsets = torch.tensor([0, 4], dtype=torch.long, device=device)
    for condition in CONDITIONS:
        adapter.evaluation_adapter = dict(contracts[condition]["adapter"])
        cache_root = (
            REPO_ROOT
            / "runs/outputs"
            / ROOTS[condition]
            / "writer_lora_cache/entries"
        )
        target_entry = f"{suite}_task_{task_id:02d}_state_000"
        target_record = read_json(cache_root / target_entry / "entry.json")
        batch_entry_ids = tuple(target_record["generation"]["batch_entry_ids"])
        requests = []
        target_position = -1
        for position, entry_id in enumerate(batch_entry_ids):
            request = read_json(cache_root / entry_id / "entry.json")["request"]
            requests.append(request)
            if entry_id == target_entry:
                target_position = position
        if target_position < 0 or any(
            re.fullmatch(r"[a-z0-9_]+", entry_id) is None
            for entry_id in batch_entry_ids
        ):
            raise RuntimeError("production Writer generation batch changed")
        inputs = [
            adapter._episode_inputs(
                suite=str(request["suite"]),
                task_id=int(request["task_id"]),
                init_state_id=int(request["init_state_id"]),
            )
            for request in requests
        ]
        rows, episode_frames, _, languages = zip(*inputs, strict=True)
        evidence[condition] = rows[target_position]
        frame_batches = tuple(
            batch for episode in episode_frames for batch in episode
        )
        frames = torch.cat(frame_batches)
        offsets = [0]
        for batch in frame_batches:
            offsets.append(offsets[-1] + int(batch.shape[0]))
        video_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        tokens, masks, spans = adapter.tokenizer(list(languages))
        lengths = video_offsets[1:] - video_offsets[:-1]
        frame_video_ids = torch.repeat_interleave(
            torch.arange(len(frame_batches), device=device), lengths
        )
        video_condition_ids = torch.repeat_interleave(
            torch.arange(len(requests), device=device),
            torch.full(
                (len(requests),), 4, dtype=torch.long, device=device
            ),
        )
        copy_task_lora_state_(
            policy, adapter.identity_state, adapter.lora_contract
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            task_anchor = writer.condition_descriptor.task_anchor(
                policy, tokens, masks, spans
            )
            video_traces = writer.condition_descriptor(
                policy,
                frames,
                frame_video_ids,
                video_offsets,
                tokens.index_select(0, video_condition_ids),
                masks.index_select(0, video_condition_ids),
                spans.index_select(0, video_condition_ids),
            )
            physical = video_traces.reshape(len(requests), 4, 20, 16, 1024)
            direction, trace_evidence = factorize_trace_evidence(physical)
            route_indices, route_weights = writer.layer_m2p.route(task_anchor)
            condition_index = target_position
            reader = None
            program = None
            for route_slot in range(route_indices.shape[1]):
                expert = writer.layer_m2p.experts[
                    int(route_indices[condition_index, route_slot])
                ]
                local_reader = expert._read_traces(
                    physical[condition_index], local_offsets
                )
                local_program = expert._axis_m2p(local_reader)
                weight = route_weights[condition_index, route_slot].to(
                    local_reader.dtype
                )
                reader = local_reader * weight if reader is None else reader + local_reader * weight
                program = local_program * weight if program is None else program + local_program * weight
            assert reader is not None and program is not None
            physical_rows.append(physical[condition_index].detach().float().cpu())
            direction_rows.append(direction[condition_index].detach().float().cpu())
            trace_evidence_rows.append(
                trace_evidence[condition_index].detach().float().cpu()
            )
            readers.append(reader.detach().float().cpu())
            programs.append(program.detach().float().cpu())
            states.append(
                {
                    name: value.detach().float().cpu()
                    for name, value in writer.decode_program(program).items()
                }
            )
            routes.append(route_indices[condition_index].detach().cpu().tolist())
            route_weights_rows.append(
                route_weights[condition_index].detach().cpu().tolist()
            )

    prepared, global_task_id = _fixed_query(
        correct, processor, suite=suite, task_id=task_id, device=device
    )
    actions = [
        _policy_action(
            adapter,
            processor,
            prepared,
            {name: value.to(device) for name, value in state.items()},
            seed=7_000 + global_task_id,
        )
        for state in states
    ]
    pairs = lora_pairs(writer)
    comparisons = {}
    for index, condition in enumerate(CONDITIONS):
        comparisons[condition] = {
            "physical_trace": tensor_metrics(
                physical_rows[0], physical_rows[index]
            ),
            "direction": tensor_metrics(direction_rows[0], direction_rows[index]),
            "trace_evidence": tensor_metrics(
                trace_evidence_rows[0], trace_evidence_rows[index]
            ),
            "reader": tensor_metrics(readers[0], readers[index]),
            "program": tensor_metrics(programs[0], programs[index]),
            "effective_ba": effective_metrics(pairs, states[0], states[index]),
            "fixed_policy_action": tensor_metrics(actions[0], actions[index]),
        }
    cached = load_file(
        str(
            REPO_ROOT
            / "runs/outputs"
            / ROOTS["correct"]
            / "writer_lora_cache/entries"
            / f"{suite}_task_{task_id:02d}_state_000/lora.safetensors"
        )
    )
    result = {
        "schema_version": "ember_sparse_semantic_expert_trace_internal_task_v1",
        "task_index": task_index,
        "global_task_id": global_task_id,
        "suite": suite,
        "task_id": task_id,
        "conditions": list(CONDITIONS),
        "route_indices": routes,
        "route_weights": route_weights_rows,
        "evidence": evidence,
        "comparisons_to_correct": comparisons,
        "correct_geometry": adapter_geometry(writer, pairs, states[0], 1.0),
        "production_replay_effective_ba": effective_metrics(
            pairs,
            states[0],
            {name: value.float() for name, value in cached.items()},
        ),
        "target_action_reads": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / f"task_{task_index:02d}.json", result)
    adapter.store.close()


def _aggregate(output_dir: Path) -> None:
    rows = [read_json(output_dir / f"task_{index:02d}.json") for index in range(8)]
    stages = tuple(rows[0]["comparisons_to_correct"]["wrong"])
    comparisons = {
        condition: {
            stage: distribution(
                [
                    float(row["comparisons_to_correct"][condition][stage]["relative_l2"])
                    for row in rows
                ]
            )
            for stage in stages
        }
        for condition in CONDITIONS[1:]
    }
    geometry_keys = (
        "effective_lora_norm_unscaled",
        "stable_rank_mean",
        "top_singular_energy_mean",
    )
    result = {
        "schema_version": "ember_sparse_semantic_expert_trace_internal_analysis_v1",
        "method_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "tasks": rows,
        "summary": {
            "comparisons_to_correct_relative_l2": comparisons,
            "correct_geometry": {
                key: distribution(
                    [float(row["correct_geometry"][key]) for row in rows]
                )
                for key in geometry_keys
            },
            "correct_top4_target_energy_fraction": distribution(
                [
                    float(
                        row["correct_geometry"]["target_energy_profile"]
                        ["top4_targets_energy_fraction"]
                    )
                    for row in rows
                ]
            ),
            "routes_identical_across_five_arms": all(
                len({tuple(route) for route in row["route_indices"]}) == 1
                for row in rows
            ),
            "maximum_production_replay_relative_l2": max(
                float(row["production_replay_effective_ba"]["relative_l2"])
                for row in rows
            ),
            "target_action_reads": 0,
        },
    }
    write_json_atomic(output_dir / "results.json", result)
    print(json.dumps(result["summary"], sort_keys=True))


def main() -> None:
    args = _args()
    if args.aggregate:
        _aggregate(args.output_dir)
    else:
        _capture_task(int(args.task_index), args.output_dir, _contracts())


if __name__ == "__main__":
    main()
