"""Fit-only cross-episode PI0.5 flow supervision for G3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch
from torch.utils.data import default_collate

from ember.ecp.natural_program_data import NaturalProgramSample
from ember.writer.data import FunctionalQueryDataset
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    task_logical_batch_policy_rng_seed,
    writer_chain_rule_surrogate,
)

if TYPE_CHECKING:
    from ember.ecp.shared_compiler_training import SharedCompilerRuntime


@dataclass(frozen=True)
class SharedCompilerFunctionalQuery:
    batch: dict[str, Any]
    demo_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]


def _functional_query(
    *,
    dataset: FunctionalQueryDataset,
    processor: Any,
    task_id: int,
    action_demos: tuple[int, ...],
    visit: int,
    seed: int,
    query_count: int,
) -> SharedCompilerFunctionalQuery:
    """Build one deterministic fit-only action panel disjoint from the videos."""

    if (
        task_id not in dataset.task_episode_rows
        or not action_demos
        or len(set(action_demos)) != len(action_demos)
        or visit < 0
        or seed < 0
        or query_count <= 0
    ):
        raise ValueError("invalid G3 cross-episode functional query")
    rows_by_demo = dataset.task_episode_rows[task_id]
    if any(demo not in rows_by_demo for demo in action_demos):
        raise ValueError("G3 functional query escaped its reserved action episodes")
    generator = np.random.default_rng(
        np.random.SeedSequence([seed, task_id, visit, 0x46334733])
    )
    selected = []
    demos = []
    frames = []
    for query in range(query_count):
        demo = int(action_demos[query % len(action_demos)])
        rows = rows_by_demo[demo]
        row = int(rows[int(generator.integers(0, len(rows)))])
        selected.append(row)
        observed_task, observed_demo, observed_frame = dataset.frame_index[row]
        if observed_task != task_id or observed_demo != demo:
            raise ValueError("G3 functional-query index changed ownership")
        demos.append(demo)
        frames.append(int(observed_frame))
    return SharedCompilerFunctionalQuery(
        batch=processor.training_batch(
            default_collate([dataset[index] for index in selected])
        ),
        demo_indices=tuple(demos),
        frame_indices=tuple(frames),
    )


def cross_episode_flow_loss(
    runtime: SharedCompilerRuntime,
    *,
    task_id: int,
    sample: NaturalProgramSample,
    macro: int,
    complete: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiate exact fit-action flow through one generated rank16 LoRA."""

    if set(sample.video_demos) & set(sample.action_demos):
        raise RuntimeError("G3 video/action episodes are not disjoint")
    query = _functional_query(
        dataset=runtime.query_dataset,
        processor=runtime.query_processor,
        task_id=task_id,
        action_demos=sample.action_demos,
        visit=macro,
        seed=int(runtime.config["optimization"]["seed"]),
        query_count=int(runtime.config["optimization"]["functional_query_count"]),
    )
    rng_seed = task_logical_batch_policy_rng_seed(
        optimization_seed=int(runtime.config["optimization"]["seed"]),
        task_id=task_id,
        task_visit=macro,
        demo_indices=query.demo_indices,
        frame_indices=query.frame_indices,
    )
    value, details, gradients = functional_lora_loss_gradient(
        runtime.policy,
        complete,
        runtime.ranks.contract,
        batch=query.batch,
        policy_rng_seed=rng_seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("G3 cross-episode PI0.5 flow loss changed")
    bridge = value.detach() + writer_chain_rule_surrogate(complete, gradients)
    return bridge, {
        "functional_action_demos": list(query.demo_indices),
        "functional_action_frames": list(query.frame_indices),
        "functional_policy_rng_seed": rng_seed,
    }
