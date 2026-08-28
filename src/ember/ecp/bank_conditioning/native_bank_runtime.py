"""Frozen G3 runtime and exact current-video candidate banks.

This module owns the reusable boundary between the frozen source/G2/compiler
stack and small G3 capacity probes.  It materializes only requested targets;
deployment implementations may replay the same interfaces chunk by chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from ember.ecp.bank_conditioning.anchor_solve import candidate_mass
from ember.ecp.bank_conditioning.mapping_eval_runtime import load_mapping_tasks
from ember.ecp.bank_conditioning.mapping_training import _load_training_assets
from ember.ecp.native_factors import NativeOutputBankState, native_output_group_count
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler_assets import authority_path, load_shared_compiler_config
from ember.ecp.shared_compiler_authority import pure_shared_compiler_inventory
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.pi05_source_setup import initialize_distributed, seed_everything


@dataclass(frozen=True)
class NativeCandidateBank:
    """One target side/group with real values and authorized candidate measures."""

    values: torch.Tensor
    content_keys: torch.Tensor
    base_mass: torch.Tensor
    event_mass: torch.Tensor
    replay_mass: torch.Tensor

    @property
    def keys(self) -> torch.Tensor:
        """Expose the S1 event view without copying event-invariant content."""

        return self.content_keys[None].expand(
            self.event_mass.shape[0], *self.content_keys.shape
        )


@dataclass
class FrozenNativeBankRuntime:
    """Frozen source, Natural Program, candidate encoder, and asset handles."""

    context: Any
    config: dict[str, Any]
    task_by_id: dict[int, Any]
    video_store: Any
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: torch.nn.Module
    compiler: torch.nn.Module
    owners: tuple[Any, ...]
    native_teachers: Any
    query_points: int
    inventory: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()


def prepare_frozen_native_bank_runtime(
    *,
    reference_config: Path,
    asset_root: Path,
    data_root: Path,
) -> FrozenNativeBankRuntime:
    """Load the single pure-Native G3 bank path with every authority frozen."""

    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1 or context.rank != 0:
        raise ValueError("native-bank probes are independent single-GPU jobs")
    config = load_shared_compiler_config(reference_config)
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = load_mapping_tasks(config, asset_root=asset_root, data_root=data_root)
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=asset_root
    )
    tokenizer = authority_path(config, "tokenizer", asset_root=asset_root)
    assets = _load_training_assets(
        SimpleNamespace(
            asset_root=asset_root,
            checkpoint=source_checkpoint,
            source_run=source_checkpoint.parent.parent,
            tokenizer_path=tokenizer,
        ),
        config,
        context,
        tasks,
    )
    inventory = pure_shared_compiler_inventory(
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        owners=assets.owners,
    )
    assets.compiler.requires_grad_(False).eval()
    if any(
        parameter.requires_grad
        for module in (assets.policy, assets.program, assets.compiler)
        for parameter in module.parameters()
    ):
        assets.video_store.close()
        raise RuntimeError("frozen native-bank runtime loaded trainable authority")
    torch.cuda.reset_peak_memory_stats(context.device)
    return FrozenNativeBankRuntime(
        context=context,
        config=config,
        task_by_id={task.authority_id: task for task in tasks},
        video_store=assets.video_store,
        language_tokens=assets.language_tokens,
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        owners=assets.owners,
        native_teachers=assets.native_teachers,
        query_points=assets.query_points,
        inventory=inventory,
    )


def prepare_k1_condition(
    runtime: FrozenNativeBankRuntime,
    *,
    task_id: int,
    video_demo: int,
    robustness_view: str,
) -> Any:
    """Capture one authorized video with frozen Pass A and real native hooks."""

    sample = NaturalProgramSample(
        video_demos=(int(video_demo),),
        action_demos=(),
        k=1,
        robustness_view=str(robustness_view),
    )
    packed = pack_shared_compiler_videos(
        task=runtime.task_by_id[int(task_id)],
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[int(task_id)]
    condition = prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=runtime.owners,
        packed=packed,
        language_tokens=tokens,
        language_mask=mask,
        chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
    )
    if len(condition.videos) != 1 or condition.metrics.get("K") != 1:
        raise RuntimeError("native-bank capture lost exact K1")
    return condition


def _projected_candidate_keys(
    scorer: Any,
    value: torch.Tensor,
    metadata: torch.Tensor,
    *,
    target: int,
    output: bool,
) -> torch.Tensor:
    if output:
        raw = scorer.output_keys(value, metadata, target=target)
        return scorer.output_projected_keys(raw, target=target)
    raw = scorer.input_keys(value, metadata, target=target)
    return scorer.input_projected_keys(raw, target=target)


def materialize_condition_banks(
    runtime: FrozenNativeBankRuntime,
    condition: Any,
    targets: Sequence[int],
) -> tuple[Any, dict[tuple[int, str, int], NativeCandidateBank]]:
    """Build exact X and dynamic-Y banks without merging their candidate axes."""

    selected = tuple(map(int, targets))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("native-bank target selection must be nonempty and unique")
    video = condition.videos[0]
    scorer = runtime.compiler.anchor_scorer
    state = scorer.program_state(condition.program)
    base_frame, event_frame, frame_measure = runtime.compiler._video_measures(
        video, state.event_weights
    )
    input_values: dict[int, list[torch.Tensor]] = {target: [] for target in selected}
    input_keys: dict[int, list[torch.Tensor]] = {target: [] for target in selected}
    output_values: dict[tuple[int, int], list[torch.Tensor]] = {}
    output_keys: dict[tuple[int, int], list[torch.Tensor]] = {}
    boundaries = {
        target: NativeOutputBankState(
            final=video.native.final_outputs[target].detach()
        )
        for target in selected
    }
    next_frame = 0
    with torch.no_grad():
        for chunk in video.native.chunks():
            stop = next_frame + chunk.frame_count
            if chunk.start_frame != next_frame or stop > video.native.frame_count:
                raise RuntimeError("native candidate chunks changed")
            assignment = video.canonical_assignment[next_frame:stop].float()
            frame_metadata = scorer.frame_metadata(
                assignment, video.frame_positions[next_frame:stop]
            )
            x_metadata = scorer.candidate_metadata(frame_metadata, output=False)
            y_metadata = scorer.candidate_metadata(frame_metadata, output=True)
            for target in selected:
                owner = runtime.owners[target]
                x = chunk.inputs[target].detach()
                input_values[target].append(x)
                key = _projected_candidate_keys(
                    scorer, x, x_metadata, target=target, output=False
                )
                input_keys[target].append(key)

                dynamic = boundaries[target].build(
                    chunk.outputs[target].detach(), start_frame=next_frame
                )
                groups = native_output_group_count(owner)
                grouped = dynamic.reshape(
                    *dynamic.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
                keys = _projected_candidate_keys(
                    scorer,
                    grouped,
                    y_metadata[None],
                    target=target,
                    output=True,
                )
                for group in range(groups):
                    output_values.setdefault((target, group), []).append(
                        grouped[group]
                    )
                    output_keys.setdefault((target, group), []).append(keys[group])
            next_frame = stop
    if next_frame != video.native.frame_count or any(
        boundary.next_frame != next_frame for boundary in boundaries.values()
    ):
        raise RuntimeError("native candidate stream ended early")

    x_base = candidate_mass(base_frame, output=False)
    x_event = candidate_mass(event_frame, output=False)
    y_base = candidate_mass(base_frame, output=True)
    y_event = candidate_mass(event_frame, output=True)
    banks: dict[tuple[int, str, int], NativeCandidateBank] = {}
    for target in selected:
        banks[(target, "input", 0)] = NativeCandidateBank(
            values=torch.cat(input_values[target], dim=0),
            content_keys=torch.cat(input_keys[target], dim=0),
            base_mass=x_base,
            event_mass=x_event,
            replay_mass=candidate_mass(frame_measure[target], output=False),
        )
        for group in range(native_output_group_count(runtime.owners[target])):
            banks[(target, "output", group)] = NativeCandidateBank(
                values=torch.cat(output_values[(target, group)], dim=0),
                content_keys=torch.cat(output_keys[(target, group)], dim=0),
                base_mass=y_base,
                event_mass=y_event,
                replay_mass=candidate_mass(frame_measure[target], output=True),
            )
    return state, banks
