"""Held-task mechanism Gate for G2 Natural Program."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ember.ecp.behavior.gate import build_behavior_gate
from ember.ecp.behavior.kernel import (
    kernel_ridge_predictions,
    program_behavior_features,
    topology_summary,
)
from ember.ecp.natural_program import NaturalProgram, NaturalProgramOutput
from ember.ecp.natural_program_data import (
    NaturalProgramSample,
    NaturalProgramTask,
    pack_natural_program_condition,
)
from ember.ecp.stage0_train_step import _gather_records
from ember.pi05_source_checkpoint import write_json_atomic

if TYPE_CHECKING:
    from ember.ecp.natural_program_training import NaturalProgramRuntime


GATE_SCHEMA = "ember_ecp_natural_program_g2_gate_v3"


def _owner_event_vector(process: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(process.float(), dim=-1)
    weighted = normalized * presence.float().clamp(0.0, 1.0).sqrt()[..., None, None]
    return weighted.flatten()


def _program_vector(output: NaturalProgramOutput) -> torch.Tensor:
    return _owner_event_vector(output.program.p_process[0], output.program.rho[0])


def _probe_vector(output: NaturalProgramOutput, probe: int) -> torch.Tensor:
    process = output.probe_process[probe].float()
    presence = output.probe_presence[probe].float()
    weights = output.alignment.float() * presence[:, None]
    mass = weights.sum(-1).clamp_min(1e-6)
    aligned = torch.einsum("vce,vejd->vcjd", weights, process) / mass[
        :, :, None, None
    ]
    aligned_presence = 1.0 - torch.exp(
        torch.log1p(
            -(output.alignment.float() * presence[:, None]).clamp(max=1.0 - 1e-6)
        ).sum(-1)
    )
    return _owner_event_vector(aligned.mean(0), aligned_presence.mean(0))


def _distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).square().mean().sqrt())


def _action_progress_loss(output: NaturalProgramOutput, batch: Any) -> dict[str, float]:
    action = F.mse_loss(
        output.predictions.action_phases.float(), batch.action_targets.float()
    )
    progress = F.mse_loss(
        output.predictions.progress.float(), batch.progress_targets.float()
    )
    return {
        "action": float(action),
        "progress": float(progress),
        "combined": float(action + progress),
    }


def _k1_exact(output: NaturalProgramOutput) -> bool:
    program = output.program
    return all(
        (
            torch.equal(program.p_scene[0], output.local_scene[0]),
            torch.equal(program.p_process[0], output.local_process[0]),
            torch.equal(program.rho[0], output.local_presence[0]),
            torch.equal(program.tau[0], output.local_tau[0]),
            torch.equal(program.sigma[0], output.local_sigma[0]),
            torch.equal(output.alignment[0], torch.eye(
                output.alignment.shape[-1],
                dtype=output.alignment.dtype,
                device=output.alignment.device,
            )),
        )
    )


def _program_difference(
    left: NaturalProgram, right: NaturalProgram
) -> tuple[float, bool]:
    differences = [
        (getattr(left, name).float() - getattr(right, name).float()).abs().max()
        for name in ("p_lang", "p_scene", "p_process", "rho", "tau", "sigma")
    ]
    maximum = float(torch.stack(differences).max())
    passed = all(
        torch.allclose(
            getattr(left, name).float(),
            getattr(right, name).float(),
            atol=1e-5,
            rtol=1e-5,
        )
        for name in ("p_lang", "p_scene", "p_process", "rho", "tau", "sigma")
    )
    return maximum, passed


def _held_assignments(
    tasks: Sequence[NaturalProgramTask], world_size: int
) -> tuple[tuple[NaturalProgramTask, ...], ...]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            -sum(task.episode_lengths[index] for index in range(19)),
            task.authority_id,
        ),
    )
    groups: list[list[NaturalProgramTask]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for task in ordered:
        rank = min(range(world_size), key=lambda row: (loads[row], row))
        groups[rank].append(task)
        loads[rank] += sum(task.episode_lengths[index] for index in range(19))
    return tuple(tuple(group) for group in groups)


def _sample(row: Mapping[str, Any]) -> NaturalProgramSample:
    return NaturalProgramSample(
        video_demos=tuple(map(int, row["video_demos"])),
        action_demos=tuple(map(int, row["action_demos"])),
        k=len(row["video_demos"]),
        robustness_view="speed2",
    )


def _forward(
    runtime: "NaturalProgramRuntime",
    task: NaturalProgramTask,
    sample: NaturalProgramSample,
    *,
    view: str,
) -> tuple[NaturalProgramOutput, Any]:
    batch = pack_natural_program_condition(
        task=task,
        sample=sample,
        video_store=runtime.video_store,
        action_store=runtime.action_store,
        label_store=runtime.label_store,
        query_points=int(runtime.config["data"]["query_points"]),
        predicate_slots=int(runtime.config["model"]["predicate_slots"]),
        device=runtime.context.device,
        view=view,
    )
    language_tokens, language_mask = runtime.language_tokens[task.authority_id]
    output = runtime.model(
        policy=runtime.policy,
        frames=batch.frames,
        frame_indices=batch.frame_indices,
        raw_frame_counts=batch.raw_frame_counts,
        video_offsets=batch.video_offsets,
        video_set_offsets=batch.video_set_offsets,
        frame_condition_ids=batch.frame_condition_ids,
        language_tokens=language_tokens,
        language_mask=language_mask,
        query_times=batch.query_times,
    )
    return output, batch


def _task_gate_record(
    runtime: "NaturalProgramRuntime",
    task: NaturalProgramTask,
) -> dict[str, Any]:
    panel = runtime.config["held_panel"]
    first = _sample(panel["same_task_sets"][0])
    second = _sample(panel["same_task_sets"][1])
    single = _sample(panel["K1_identity"])
    multiple = _sample(panel["K4_permutation"])
    permutation = tuple(map(int, panel["K4_permutation"]["permutation"]))
    permuted = NaturalProgramSample(
        video_demos=tuple(multiple.video_demos[index] for index in permutation),
        action_demos=tuple(multiple.action_demos[index] for index in permutation),
        k=multiple.k,
        robustness_view=multiple.robustness_view,
    )

    output_a, batch_a = _forward(runtime, task, first, view="full")
    endpoint_a, endpoint_batch_a = _forward(runtime, task, first, view="endpoints")
    output_b, batch_b = _forward(runtime, task, second, view="full")
    endpoint_b, endpoint_batch_b = _forward(runtime, task, second, view="endpoints")
    output_k1, _ = _forward(runtime, task, single, view="full")
    output_k4, _ = _forward(runtime, task, multiple, view="full")
    output_permuted, _ = _forward(runtime, task, permuted, view="full")
    permutation_max_abs, permutation_pass = _program_difference(
        output_k4.program, output_permuted.program
    )

    active_a = int((output_a.program.rho[0].float() > 0.5).sum())
    active_b = int((output_b.program.rho[0].float() > 0.5).sum())
    if runtime.behavior_codes is None:
        raise RuntimeError("G2 held Gate lost behavior-kernel authority")
    behavior_features = {
        name: program_behavior_features(
            output.program, runtime.behavior_codes.selected_targets
        )[0].detach().cpu()
        for name, output in (
            ("same_a", output_a),
            ("same_b", output_b),
            ("k1", output_k1),
            ("k4", output_k4),
        )
    }
    role = "meta_held" if task.authority_id < 71 else "target_held"
    return {
        "authority_id": task.authority_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "role": role,
        "embedding_a": _program_vector(output_a).detach().cpu(),
        "embedding_b": _program_vector(output_b).detach().cpu(),
        "probe_delta_a": _distance(
            _probe_vector(output_a, 0), _probe_vector(output_a, 1)
        ),
        "probe_delta_b": _distance(
            _probe_vector(output_b, 0), _probe_vector(output_b, 1)
        ),
        "active_events": [active_a, active_b],
        "one_event_rows": int(active_a <= 1) + int(active_b <= 1),
        "full_losses": [
            _action_progress_loss(output_a, batch_a),
            _action_progress_loss(output_b, batch_b),
        ],
        "endpoint_losses": [
            _action_progress_loss(endpoint_a, endpoint_batch_a),
            _action_progress_loss(endpoint_b, endpoint_batch_b),
        ],
        "K1_exact_identity": _k1_exact(output_k1),
        "K4_permutation_max_abs": permutation_max_abs,
        "K4_permutation_invariant": permutation_pass,
        "mean_sigma": float(
            torch.stack(
                (output_a.program.sigma.float().mean(), output_b.program.sigma.float().mean())
            ).mean()
        ),
        "tau_order_violation_fraction": float(
            torch.cat(
                (
                    output_a.program.tau[:, 1:, 0]
                    < output_a.program.tau[:, :-1, 0],
                    output_b.program.tau[:, 1:, 0]
                    < output_b.program.tau[:, :-1, 0],
                ),
                dim=0,
            ).float().mean()
        ),
        "behavior_features": behavior_features,
        "behavior_predictions": None,
    }


def _task_kernel_record(
    runtime: "NaturalProgramRuntime", task: NaturalProgramTask
) -> dict[str, Any]:
    panel = runtime.config["held_panel"]
    outputs = [
        _forward(runtime, task, _sample(row), view="full")[0]
        for row in panel["same_task_sets"]
    ]
    if runtime.behavior_codes is None:
        raise RuntimeError("G2 topology Gate lost behavior authority")
    return {
        "authority_id": task.authority_id,
        "role": "meta_fit" if task.authority_id < 71 else "target_fit",
        "behavior_features": {
            name: program_behavior_features(
                output.program, runtime.behavior_codes.selected_targets
            )[0].detach().cpu()
            for name, output in zip(("same_a", "same_b"), outputs, strict=True)
        },
    }


def _behavior_kernel_qualification(
    runtime: "NaturalProgramRuntime",
    train_records: list[dict[str, Any]],
    held_records: list[dict[str, Any]],
) -> dict[str, Any]:
    authority = runtime.behavior_codes
    if authority is None:
        raise RuntimeError("G2 behavior-kernel qualification lost its authority")
    train_records.sort(key=lambda row: int(row["authority_id"]))
    held_records.sort(key=lambda row: int(row["authority_id"]))
    device = runtime.context.device
    train_ids = torch.tensor(
        [int(row["authority_id"]) for row in train_records],
        dtype=torch.long,
        device=device,
    )
    held_ids = torch.tensor(
        [int(row["authority_id"]) for row in held_records],
        dtype=torch.long,
        device=device,
    )

    def features(records: Sequence[Mapping[str, Any]], view: str) -> torch.Tensor:
        return torch.stack(
            [row["behavior_features"][view].to(device) for row in records]
        ).float()

    train_a = features(train_records, "same_a")
    train_b = features(train_records, "same_b")
    held_a = features(held_records, "same_a")
    held_b = features(held_records, "same_b")
    topology = {
        "train": topology_summary(
            features_a=train_a,
            features_b=train_b,
            task_ids=train_ids,
            authority=authority,
            roles={
                "meta_fit": authority.meta_gradient_task_ids,
                "target_fit": authority.target_gradient_task_ids,
            },
        ),
        "held": topology_summary(
            features_a=held_a,
            features_b=held_b,
            task_ids=held_ids,
            authority=authority,
            roles={
                "meta_held": tuple(task for task in authority.held_task_ids if task < 71),
                "target_held": tuple(task for task in authority.held_task_ids if task >= 71),
            },
        ),
    }
    targets = torch.stack(
        [authority.target(int(task), standardized=True) for task in train_ids]
    )
    fit_features = torch.cat((train_a, train_b), dim=0)
    fit_targets = torch.cat((targets, targets), dim=0)
    ridge = float(runtime.config["behavior_alignment"]["evaluator_ridge"])
    for view in ("same_a", "same_b", "k1", "k4"):
        query = features(held_records, view)
        standardized = kernel_ridge_predictions(
            train_features=fit_features,
            train_targets=fit_targets,
            query_features=query,
            ridge=ridge,
        )
        decoded = authority.decode(standardized).detach().cpu()
        for row, prediction in zip(held_records, decoded, strict=True):
            if row["behavior_predictions"] is None:
                row["behavior_predictions"] = {}
            row["behavior_predictions"][view] = prediction
    return topology


def _distance_rows(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[bool], list[bool]]:
    pair_passes = []
    probe_passes = []
    task_rows = []
    embeddings_a = [row["embedding_a"] for row in records]
    embeddings_b = [row["embedding_b"] for row in records]
    for index, row in enumerate(records):
        embedding_a = embeddings_a[index]
        embedding_b = embeddings_b[index]
        same = _distance(embedding_a, embedding_b)
        cross_a = min(
            _distance(embedding_a, other)
            for other_index, other in enumerate(embeddings_b)
            if other_index != index
        )
        cross_b = min(
            _distance(embedding_b, other)
            for other_index, other in enumerate(embeddings_a)
            if other_index != index
        )
        pair_pass = same < min(cross_a, cross_b)
        probe_a_pass = float(row["probe_delta_a"]) < 0.5 * cross_a
        probe_b_pass = float(row["probe_delta_b"]) < 0.5 * cross_b
        pair_passes.append(pair_pass)
        probe_passes.extend((probe_a_pass, probe_b_pass))
        task_rows.append(
            {
                **{
                    name: value
                    for name, value in row.items()
                    if name
                    not in {"embedding_a", "embedding_b", "behavior_predictions"}
                    and name != "behavior_features"
                },
                "same_task_distance": same,
                "nearest_cross_distance_a": cross_a,
                "nearest_cross_distance_b": cross_b,
                "same_task_nearer": pair_pass,
                "probe_below_half_cross_a": probe_a_pass,
                "probe_below_half_cross_b": probe_b_pass,
            }
        )
    return task_rows, pair_passes, probe_passes


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    pair_passes: Sequence[bool],
    probe_passes: Sequence[bool],
) -> dict[str, float]:
    active = [value for row in records for value in row["active_events"]]
    full = [
        loss["combined"] for row in records for loss in row["full_losses"]
    ]
    endpoints = [
        loss["combined"] for row in records for loss in row["endpoint_losses"]
    ]
    full_mean = float(np.mean(full))
    endpoint_mean = float(np.mean(endpoints))
    improvement = (endpoint_mean - full_mean) / max(endpoint_mean, 1e-12)
    return {
        "same_task_nearer_fraction": float(np.mean(pair_passes)),
        "probe_delta_below_half_cross_margin_fraction": float(
            np.mean(probe_passes)
        ),
        "one_event_fraction": float(
            sum(int(row["one_event_rows"]) for row in records) / (2 * len(records))
        ),
        "median_active_events": float(np.median(active)),
        "full_action_progress_loss": full_mean,
        "endpoints_action_progress_loss": endpoint_mean,
        "full_vs_endpoints_action_progress_improvement": improvement,
        "K1_exact_identity_fraction": float(
            np.mean([row["K1_exact_identity"] for row in records])
        ),
        "K4_permutation_invariance_fraction": float(
            np.mean([row["K4_permutation_invariant"] for row in records])
        ),
        "K4_permutation_max_abs": max(
            float(row["K4_permutation_max_abs"]) for row in records
        ),
        "mean_sigma": float(np.mean([row["mean_sigma"] for row in records])),
        "tau_order_violation_fraction": float(
            np.mean([row["tau_order_violation_fraction"] for row in records])
        ),
    }


def _threshold_checks(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "same_task_separation": metrics["same_task_nearer_fraction"]
        >= float(thresholds["same_task_nearer_fraction"]),
        "probe_stability": metrics[
            "probe_delta_below_half_cross_margin_fraction"
        ]
        >= float(thresholds["probe_delta_below_half_cross_margin_fraction"]),
        "one_event_noncollapse": metrics["one_event_fraction"]
        <= float(thresholds["maximum_one_event_fraction"]),
        "active_event_range": float(thresholds["median_active_events_min"])
        <= metrics["median_active_events"]
        <= float(thresholds["median_active_events_max"]),
        "full_beats_endpoints": metrics[
            "full_vs_endpoints_action_progress_improvement"
        ]
        >= float(thresholds["full_vs_endpoints_action_progress_improvement"]),
        "K1_exact_identity": metrics["K1_exact_identity_fraction"] == 1.0,
        "K_permutation_invariance": metrics[
            "K4_permutation_invariance_fraction"
        ]
        == 1.0,
    }


def _build_report(
    records: list[dict[str, Any]],
    *,
    macro: int,
    thresholds: Mapping[str, Any],
    behavior_authority: Any | None = None,
    behavior_thresholds: Mapping[str, float] | None = None,
    behavior_topology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records.sort(key=lambda row: int(row["authority_id"]))
    expected_count = (
        len(behavior_authority.held_task_ids)
        if behavior_authority is not None
        else 20
    )
    if len(records) != expected_count or {row["role"] for row in records} != {
        "meta_held",
        "target_held",
    }:
        raise ValueError("G2 held Gate panel changed")
    task_rows, pair_passes, probe_passes = _distance_rows(records)
    metrics = _aggregate_metrics(
        records,
        pair_passes=pair_passes,
        probe_passes=probe_passes,
    )
    checks = _threshold_checks(metrics, thresholds)
    behavior = None
    if behavior_authority is not None:
        if behavior_thresholds is None:
            raise ValueError("G2 behavior Gate lost its thresholds")
        behavior = build_behavior_gate(
            records, behavior_authority, behavior_thresholds
        )
        checks["behavior_alignment"] = bool(behavior["passed"])
        if behavior_topology is None:
            raise ValueError("G2 behavior Gate lost its topology report")
        train_topology = behavior_topology["train"]
        held_topology = behavior_topology["held"]
        train_correlation = 0.5 * (
            float(train_topology["role_equal_program_to_behavior_a"])
            + float(train_topology["role_equal_program_to_behavior_b"])
        )
        held_correlation = 0.5 * (
            float(held_topology["role_equal_program_to_behavior_a"])
            + float(held_topology["role_equal_program_to_behavior_b"])
        )
        checks["train_behavior_topology"] = train_correlation >= float(
            behavior_thresholds["train_pairwise_correlation"]
        )
        checks["held_behavior_topology"] = held_correlation >= float(
            behavior_thresholds["held_pairwise_correlation"]
        )
        checks["held_role_behavior_topology"] = all(
            0.5
            * (
                float(value["program_to_behavior_a"])
                + float(value["program_to_behavior_b"])
            )
            > float(behavior_thresholds["held_role_minimum_correlation"])
            for value in held_topology["by_role"].values()
        )
    return {
        "schema_version": GATE_SCHEMA,
        "stage": "g2_natural_program",
        "checkpoint_macro": int(macro),
        "panel": {
            "tasks": len(records),
            "meta_held": sum(row["role"] == "meta_held" for row in records),
            "target_held": sum(row["role"] == "target_held" for row in records),
            "gradient_updates": 0,
            "shuffled_or_reversed_conditions": 0,
        },
        "distance_definition": (
            "RMS over canonical owner/event P_process unit vectors weighted by "
            "sqrt(rho); same pair must beat both directional nearest cross tasks"
        ),
        "probe_definition": (
            "antithetic local Programs aligned by the Pass-A canonical assignment; "
            "each delta is compared with half its directional nearest-cross distance"
        ),
        "thresholds": dict(thresholds),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "behavior_alignment": behavior,
        "behavior_topology": behavior_topology,
        "tasks": task_rows,
    }


def evaluate_natural_program_gate(
    runtime: "NaturalProgramRuntime", macro: int
) -> dict[str, Any]:
    if runtime.behavior_codes is None:
        held = tuple(
            task
            for task in runtime.tasks
            if task.role in {"meta_held", "target_held"}
        )
        train: tuple[NaturalProgramTask, ...] = ()
    else:
        held_ids = set(runtime.behavior_codes.held_task_ids)
        train_ids = set(runtime.behavior_codes.fit_task_ids)
        held = tuple(task for task in runtime.tasks if task.authority_id in held_ids)
        train = tuple(task for task in runtime.tasks if task.authority_id in train_ids)
    held_groups = _held_assignments(held, runtime.context.world_size)
    train_groups = _held_assignments(train, runtime.context.world_size) if train else ()
    runtime.model.eval()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            local = [
                _task_gate_record(runtime, task)
                for task in held_groups[runtime.context.rank]
            ]
            local_train = (
                [
                    _task_kernel_record(runtime, task)
                    for task in train_groups[runtime.context.rank]
                ]
                if train_groups
                else []
            )
        records = _gather_records(local, runtime.context.world_size)
        train_records = _gather_records(local_train, runtime.context.world_size)
        payload: list[Any] = [None]
        if runtime.context.is_main:
            try:
                topology = (
                    _behavior_kernel_qualification(
                        runtime, train_records, records
                    )
                    if runtime.behavior_codes is not None
                    else None
                )
                report = _build_report(
                    records,
                    macro=macro,
                    thresholds=runtime.config["gate"],
                    behavior_authority=runtime.behavior_codes,
                    behavior_thresholds=(
                        runtime.config["behavior_alignment"]["gate"]
                        if runtime.behavior_codes is not None
                        else None
                    ),
                    behavior_topology=topology,
                )
                path = runtime.args.output_dir / "gates" / f"macro_{macro:08d}.json"
                if path.exists():
                    raise ValueError(f"G2 Gate report already exists: {path}")
                write_json_atomic(path, report)
                payload[0] = report
            except Exception as error:
                payload[0] = {"error": repr(error)}
        if runtime.context.world_size > 1:
            dist.broadcast_object_list(payload, src=0, device=runtime.context.device)
        if payload[0].get("error"):
            raise ValueError(payload[0]["error"])
        return payload[0]
    finally:
        runtime.model.train()
        runtime.model.encoder.eval()
