#!/usr/bin/env python3
"""Audit real functional gradients for every intended Writer module group."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.writer.as_config import REPO_ROOT
from ember.writer.as_step import (
    _batch_task_id,
    _pack_condition,
    _policy_seed,
    _task_gradient,
    _teacher_demos,
)
from ember.writer.training import (
    _build_sampler,
    build_parser as build_training_parser,
    finalize_args,
    prepare_runtime,
)


FACTOR_FAMILIES = (
    "q_a",
    "q_b",
    "v_a",
    "v_b",
    "action_in_a",
    "action_in_b",
    "action_out_a",
    "action_out_b",
)


def _group(name: str) -> str:
    exact_prefixes = (
        ("semantic_encoder.patch_grounding.query.", "patch_grounding.query"),
        ("semantic_encoder.patch_grounding.key.", "patch_grounding.key"),
        ("semantic_encoder.patch_grounding.output.", "patch_grounding.output"),
        ("semantic_encoder.patch_grounding.", "patch_grounding.norm"),
        ("semantic_encoder.interaction_projection.", "interaction_projection"),
        ("semantic_encoder.language_projection.", "language_projection"),
        ("semantic_encoder.action_meta_lora.", "action_meta_lora"),
        ("semantic_core.", "semantic_core"),
        ("visual_transition.", "visual_transition"),
        ("procedure.", "procedure"),
        ("memory_reader.", "memory_reader"),
        ("video_set.", "video_set"),
        ("compiler.core_fusion.", "core_fusion"),
        ("compiler.", "m2p"),
    )
    if name == "memory_tokens":
        return "memory_tokens"
    for family in FACTOR_FAMILIES:
        if name.startswith(f"factor_heads.{family}."):
            return f"factor_heads.{family}"
    for prefix, group in exact_prefixes:
        if name.startswith(prefix):
            return group
    return "unclassified"


def _record_hooks(
    writer: torch.nn.Module,
) -> tuple[dict[str, dict[str, Any]], list[torch.utils.hooks.RemovableHandle]]:
    records: dict[str, dict[str, Any]] = {}
    handles = []
    for name, parameter in writer.named_parameters():
        if not parameter.requires_grad:
            continue
        record = {
            "group": _group(name),
            "parameter_count": parameter.numel(),
            "gradient_present": False,
            "gradient_nonzero": False,
            "gradient_finite": None,
            "gradient_l2": 0.0,
        }
        records[name] = record

        def observe(
            gradient: torch.Tensor, *, destination: dict[str, Any] = record
        ) -> torch.Tensor:
            value = gradient.detach().float()
            destination["gradient_present"] = True
            destination["gradient_nonzero"] = bool(value.count_nonzero())
            destination["gradient_finite"] = bool(torch.isfinite(value).all())
            destination["gradient_l2"] = float(torch.linalg.vector_norm(value))
            return gradient

        handles.append(parameter.register_hook(observe))
    return records, handles


def _aggregate(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, dict[str, Any]] = {}
    for record in records.values():
        group = str(record["group"])
        row = output.setdefault(
            group,
            {
                "parameter_tensors": 0,
                "parameter_count": 0,
                "gradient_present_tensors": 0,
                "gradient_nonzero_tensors": 0,
                "all_present_gradients_finite": True,
                "gradient_l2": 0.0,
            },
        )
        row["parameter_tensors"] += 1
        row["parameter_count"] += int(record["parameter_count"])
        row["gradient_present_tensors"] += bool(record["gradient_present"])
        row["gradient_nonzero_tensors"] += bool(record["gradient_nonzero"])
        if record["gradient_present"]:
            row["all_present_gradients_finite"] &= bool(record["gradient_finite"])
        row["gradient_l2"] += float(record["gradient_l2"]) ** 2
    for row in output.values():
        row["gradient_l2"] = math.sqrt(float(row["gradient_l2"]))
    return dict(sorted(output.items()))


def _reset_schedule(runtime: Any, macro: int) -> None:
    sampler, schedule, loader = _build_sampler(
        dataset=runtime.dataset,
        task_ids=runtime.task_ids,
        config=runtime.config,
        video_data=runtime.contract["video_data"],
        context=runtime.context,
        start_macro=macro,
        stop_macro=macro + 1,
        batch_size=20,
        num_workers=0,
    )
    runtime.sampler = sampler
    runtime.video_schedule = schedule
    runtime.iterator = iter(loader)


def _audit_state(runtime: Any, *, label: str, macro: int) -> dict[str, Any]:
    _reset_schedule(runtime, macro)
    task_id, task_visit = runtime.sampler.task_visit_for_step(macro, 0)
    batch = next(runtime.iterator)
    if _batch_task_id(batch) != task_id:
        raise RuntimeError("gradient audit sampler and batch task disagree")
    demos = _teacher_demos(
        runtime, task_id=task_id, task_visit=task_visit, batch=batch
    )
    packed, video_metrics = _pack_condition(runtime, task_id, demos)
    policy_seed = _policy_seed(runtime, batch, task_id, task_visit)
    records, handles = _record_hooks(runtime.writer)
    flat = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    try:
        loss, detail = _task_gradient(
            runtime,
            packed,
            runtime.processor.training_batch(batch),
            policy_seed,
            flat,
        )
    finally:
        for handle in handles:
            handle.remove()
    source_gradients = [
        parameter.grad
        for parameter in runtime.policy.parameters()
        if parameter.grad is not None
    ]
    grouped = _aggregate(records)
    return {
        "label": label,
        "schedule_macro": macro,
        "task_id": task_id,
        "task_visit": task_visit,
        "teacher_demo_indices": list(demos),
        "policy_rng_seed": policy_seed,
        "functional_loss": float(loss),
        "functional_detail": {
            key: value
            for key, value in detail.items()
            if isinstance(value, (bool, float, int, str))
        },
        "video_metrics": video_metrics,
        "groups": grouped,
        "unclassified_parameter_names": sorted(
            name for name, record in records.items() if record["group"] == "unclassified"
        ),
        "source_policy_gradient_tensors": len(source_gradients),
        "source_policy_nonzero_gradient_tensors": sum(
            bool(value.detach().count_nonzero()) for value in source_gradients
        ),
    }


def _first_nonzero(states: list[Mapping[str, Any]]) -> dict[str, str | None]:
    groups = sorted({group for state in states for group in state["groups"]})
    return {
        group: next(
            (
                str(state["label"])
                for state in states
                if state["groups"].get(group, {}).get(
                    "gradient_nonzero_tensors", 0
                )
            ),
            None,
        )
        for group in groups
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_training_parser()
    parser.description = __doc__
    parser.add_argument(
        "--audit-state",
        action="append",
        type=Path,
        default=[],
        help="Writer safetensors loaded after the fresh audit",
    )
    parser.add_argument("--audit-macro", type=int, default=25)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    args.audit_state = [path.resolve() for path in args.audit_state]
    args.audit_output = args.audit_output.resolve()
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    runtime = None
    try:
        runtime = prepare_runtime(args, context)
        states = [_audit_state(runtime, label="fresh", macro=args.audit_macro)]
        for path in args.audit_state:
            runtime.writer.load_state_dict(
                load_file(str(path), device=str(context.device)), strict=True
            )
            states.append(
                _audit_state(
                    runtime,
                    label=path.parent.name,
                    macro=args.audit_macro,
                )
            )
        repository = git_state(REPO_ROOT)
        output = {
            "schema_version": "ember_writer_functional_gradient_audit_v1",
            "repository": {
                "commit": repository["commit"],
                "dirty_paths": repository["dirty_paths"],
            },
            "states": states,
            "first_observed_nonzero_state": _first_nonzero(states),
        }
        write_json_atomic(args.audit_output, output)
        print(json.dumps(output, sort_keys=True), flush=True)
    finally:
        if runtime is not None:
            runtime.dataset.close()
            runtime.video_store.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
