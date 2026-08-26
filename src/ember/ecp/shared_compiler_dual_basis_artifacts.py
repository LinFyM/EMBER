"""Artifact IO and task-balanced summaries for the G3 dual-basis probe."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.shared_compiler_dual_basis import (
    BASIS_SCHEMA,
    DEFAULT_BASIS_DIMENSIONS,
    DEFAULT_TARGETS,
    REPLAY_SCHEMA,
    REPORT_SCHEMA,
    WORKER_SCHEMA,
    quantiles as _quantiles,
    probe_thresholds as _probe_thresholds,
    side_tensor_blocks as _side_tensor_rows,
    task_distribution as _task_distribution,
    task_equal_scatter as _task_equal_scatter,
)
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def save_safetensors_atomic(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def validate_formal_arguments(args: Any, *, repo_root: Path) -> None:
    if (
        args.task_ids is not None
        or not 1 <= int(args.shard_count) <= 6
        or tuple(map(int, args.target_indices)) != DEFAULT_TARGETS
        or int(args.max_videos_per_task) != 2
        or tuple(map(int, args.basis_dimensions)) != DEFAULT_BASIS_DIMENSIONS
        or abs(float(args.score_bound) - 0.1) > 1e-12
        or args.config != (repo_root / "configs/pi05_ecp_shared_compiler_g3_v2.json")
        or args.g1_config != (repo_root / "configs/pi05_ecp_native_factor_g1_v1.json")
    ):
        raise ValueError("formal dual-basis command changed its canonical contract")


def validate_formal_git(repo_root: Path, *, expected_commit: str) -> Mapping[str, Any]:
    state = git_state(repo_root)
    if (
        state.get("dirty_paths")
        or state.get("authority_contains_commit") is not True
        or state.get("commit") != expected_commit
    ):
        raise ValueError("formal dual-basis command was not clean frozen authority")
    return state


def _validate_capture_tensors(
    completion: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    expected_keys: set[str] = set()
    for record in completion["records"]:
        record_index = int(record["record_index"])
        members = int(record["member_count"])
        if members != len(record["member_names"]) or members <= 0:
            raise ValueError("dual-basis capture member inventory changed")
        targets = tuple(map(int, completion["target_indices"]))
        target_rows = tuple(record["targets"])
        if tuple(int(row["target"]) for row in target_rows) != targets:
            raise ValueError("dual-basis capture target inventory changed")
        for target_row in target_rows:
            target = int(target_row["target"])
            prefix = f"record/{record_index:06d}/target/{target:02d}"
            groups = int(target_row["groups"])
            sides = ((f"{prefix}/input", int(target_row["input_width"])),)
            sides += tuple(
                (f"{prefix}/output/g{group:03d}", int(target_row["group_width"]))
                for group in range(groups)
            )
            for base, width in sides:
                unit_key = f"{base}/unit"
                norm_key = f"{base}/dual_l2_norm"
                projection_key = f"{base}/projection_cosine"
                expected_keys.update((unit_key, norm_key, projection_key))
                unit = tensors.get(unit_key)
                norm = tensors.get(norm_key)
                projection = tensors.get(projection_key)
                if (
                    unit is None
                    or norm is None
                    or projection is None
                    or unit.shape != (members, 4, width)
                    or norm.shape != (members, 4)
                    or projection.shape != (members, 4)
                    or unit.dtype != torch.float64
                    or norm.dtype != torch.float64
                ):
                    raise ValueError("dual-basis capture tensor contract changed")
    if set(tensors) != expected_keys:
        raise ValueError("dual-basis capture tensor key inventory changed")


def capture_workers(
    output_dir: Path, shard_count: int
) -> tuple[tuple[Path, Mapping[str, Any], Mapping[str, torch.Tensor]], ...]:
    rows = []
    seen_tasks: set[int] = set()
    for shard_index in range(shard_count):
        worker_dir = output_dir / "workers" / f"worker_{shard_index:03d}"
        completion_path = worker_dir / "completion.json"
        tensor_path = worker_dir / "duals.safetensors"
        if not completion_path.is_file() or not tensor_path.is_file():
            raise ValueError(f"dual-basis capture shard {shard_index} is incomplete")
        completion = read_json(completion_path)
        if (
            completion.get("schema_version") != WORKER_SCHEMA
            or completion.get("status") != "complete"
            or completion.get("shard_index") != shard_index
            or completion.get("shard_count") != shard_count
        ):
            raise ValueError("dual-basis capture shard contract changed")
        records = tuple(completion.get("records", ()))
        record_tasks = {int(row["authority_id"]) for row in records}
        task_ids = set(map(int, completion["task_ids"]))
        tensor_file = completion.get("tensor_file", {})
        if (
            completion.get("record_count") != len(records)
            or record_tasks != task_ids
            or sorted(int(row["record_index"]) for row in records)
            != list(range(len(records)))
            or Path(str(tensor_file.get("path"))).resolve() != tensor_path.resolve()
            or tensor_file.get("bytes") != tensor_path.stat().st_size
        ):
            raise ValueError("dual-basis capture artifact inventory changed")
        if seen_tasks & task_ids:
            raise ValueError("dual-basis capture duplicated a task across workers")
        seen_tasks.update(task_ids)
        tensors = load_file(str(tensor_path))
        _validate_capture_tensors(completion, tensors)
        rows.append((worker_dir, completion, tensors))
    return tuple(rows)


def aggregate(args: Any, *, repo_root: Path) -> None:
    """Fit target-specific, task-LOTO bases without reading held authorities."""

    manifest_path = args.output_dir / "basis_manifest.json"
    basis_path = args.output_dir / "bases.safetensors"
    if manifest_path.exists() or basis_path.exists():
        raise ValueError("dual-basis aggregate output already exists")
    captures = capture_workers(args.output_dir, args.shard_count)
    dimensions = tuple(sorted(set(map(int, args.basis_dimensions))))
    if not dimensions or min(dimensions) <= 0:
        raise ValueError("dual-basis dimensions must be positive")

    target_indices: tuple[int, ...] | None = None
    dirty_subset: bool | None = None
    selection: Mapping[str, Any] | None = None
    singular_threshold: float | None = None
    task_rows: dict[tuple[int, str], dict[int, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    target_metadata: dict[int, dict[str, Any]] = {}
    target_diagnostics: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_roles: dict[int, str] = {}
    source_commits: set[str] = set()
    record_inventory = []
    source_records = 0
    for _worker_dir, completion, tensors in captures:
        current_targets = tuple(map(int, completion["target_indices"]))
        if target_indices is None:
            target_indices = current_targets
        elif target_indices != current_targets:
            raise ValueError("dual-basis workers used different target subsets")
        current_dirty = bool(completion["selection"]["dirty_exploratory_subset"])
        if dirty_subset is None:
            dirty_subset = current_dirty
        elif dirty_subset != current_dirty:
            raise ValueError("dual-basis workers mixed smoke and formal captures")
        current_selection = completion["selection"]
        if selection is None:
            selection = current_selection
        elif selection != current_selection:
            raise ValueError("dual-basis workers used different video selections")
        current_threshold = float(completion["relative_singular_threshold"])
        if singular_threshold is None:
            singular_threshold = current_threshold
        elif abs(singular_threshold - current_threshold) > 1e-15:
            raise ValueError("dual-basis workers used different solver thresholds")
        wall = completion.get("information_wall", {})
        if (
            wall.get("held_authority_reads") != 0
            or wall.get("action_meta_installed") is not False
            or wall.get("action_meta_module_count") != 0
            or wall.get("action_meta_parameter_count") != 0
            or wall.get("source_policy_trainable_parameter_count") != 0
            or wall.get("natural_program_trainable_parameter_count") != 0
            or wall.get("deployment_use") is not False
        ):
            raise ValueError("dual-basis worker crossed the information wall")
        source_git = completion.get("git", {})
        source_commits.add(str(source_git.get("commit")))
        if not current_dirty and (
            source_git.get("dirty_paths")
            or source_git.get("authority_contains_commit") is not True
        ):
            raise ValueError("formal dual-basis worker was not clean pushed main")
        for record in completion["records"]:
            task_id = int(record["authority_id"])
            role = str(record["role"])
            if role not in {"meta_fit", "target_fit"}:
                raise ValueError("dual-basis aggregate crossed fit-only authority")
            if task_id in task_roles and task_roles[task_id] != role:
                raise ValueError("dual-basis task role changed")
            task_roles[task_id] = role
            record_inventory.append(
                {
                    "authority_id": task_id,
                    "role": role,
                    "video_demo": int(record["video_demo"]),
                    "member_names": list(record["member_names"]),
                }
            )
            for target_row in record["targets"]:
                target = int(target_row["target"])
                metadata = {
                    key: target_row[key]
                    for key in (
                        "target",
                        "target_name",
                        "family",
                        "groups",
                        "input_width",
                        "output_width",
                        "group_width",
                    )
                }
                if target in target_metadata and target_metadata[target] != metadata:
                    raise ValueError("dual-basis target metadata changed")
                target_metadata[target] = metadata
                diagnostics = target_diagnostics[target]
                diagnostics["input_condition"].append(
                    float(target_row["input"]["condition"])
                )
                diagnostics["input_dual_norm_median"].append(
                    float(target_row["input"]["dual_norm_median"])
                )
                for output_row in target_row["output"]:
                    diagnostics["output_condition"].append(
                        float(output_row["condition"])
                    )
                    diagnostics["output_dual_norm_median"].append(
                        float(output_row["dual_norm_median"])
                    )
                gauge = target_row.get("canonical_rank_gauge")
                if gauge is not None:
                    diagnostics["rank_adjacent_gap_minimum"].append(
                        float(gauge["minimum_relative_adjacent_gap"])
                    )
                    diagnostics["rank_near_degenerate_fraction"].append(
                        float(gauge["near_degenerate_adjacent_fraction_at_0.05"])
                    )
                for side in ("input", "output"):
                    task_rows[(target, side)][task_id].extend(
                        _side_tensor_rows(
                            tensors,
                            record_index=int(record["record_index"]),
                            target_row=target_row,
                            side=side,
                        )
                    )
            source_records += 1
    tasks = tuple(sorted(task_roles))
    if target_indices is None or len(tasks) < 2:
        raise ValueError("dual-basis LOTO requires at least two captured tasks")
    if selection is None or singular_threshold is None:
        raise ValueError("dual-basis capture metadata disappeared")
    if len(source_commits) != 1:
        raise ValueError("dual-basis workers used different source commits")
    source_commit = next(iter(source_commits))
    if not dirty_subset:
        validate_formal_arguments(args, repo_root=repo_root)
        aggregate_git = validate_formal_git(
            repo_root, expected_commit=source_commit
        )
    else:
        aggregate_git = git_state(repo_root)
    if not dirty_subset and (
        len(tasks) != 50
        or {
            role: sum(value == role for value in task_roles.values())
            for role in ("meta_fit", "target_fit")
        }
        != {"meta_fit": 31, "target_fit": 19}
        or target_indices != DEFAULT_TARGETS
        or dimensions != DEFAULT_BASIS_DIMENSIONS
        or selection.get("selected_task_count") != 50
        or selection.get("selected_unique_video_count") != 100
        or selection.get("max_videos_per_task") != 2
        or selection.get("explicit_task_ids") is not None
    ):
        raise ValueError("formal dual-basis aggregate lost the K1-covered50 authority")
    videos_per_task: dict[int, set[int]] = defaultdict(set)
    for row in record_inventory:
        task = int(row["authority_id"])
        video = int(row["video_demo"])
        if video in videos_per_task[task]:
            raise ValueError("dual-basis capture duplicated a task video")
        videos_per_task[task].add(video)
    if not dirty_subset and (
        len(record_inventory) != 100
        or any(len(videos_per_task[task]) != 2 for task in tasks)
        or {target_metadata[target]["family"] for target in target_indices}
        != {"q", "v", "action_in", "action_out"}
    ):
        raise ValueError("formal dual-basis capture lost video or family coverage")

    device = torch.device(args.aggregate_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("dual-basis aggregate requested unavailable CUDA")
    basis_tensors: dict[str, torch.Tensor] = {}
    fits = []
    with torch.no_grad():
        for target in target_indices:
            for side in ("input", "output"):
                by_task = task_rows[(target, side)]
                if set(by_task) != set(tasks):
                    raise ValueError("dual-basis side lost task coverage")
                scatters = {
                    task: _task_equal_scatter(by_task[task]).to(device)
                    for task in tasks
                }
                widths = {value.shape[0] for value in scatters.values()}
                if len(widths) != 1:
                    raise ValueError("dual-basis native width changed across tasks")
                total = torch.stack(tuple(scatters.values())).sum(0)
                for held_task in tasks:
                    loto = (total - scatters[held_task]) / (len(tasks) - 1)
                    eigenvalues, eigenvectors = torch.linalg.eigh(loto)
                    eigenvalues = eigenvalues.clamp_min(0).flip(0)
                    eigenvectors = eigenvectors.flip(1)
                    positive = int(
                        (
                            eigenvalues
                            > eigenvalues[0].clamp_min(1e-30) * 1e-6
                        ).sum()
                    )
                    retained = min(max(dimensions), positive, eigenvectors.shape[1])
                    if retained <= 0:
                        raise ValueError("dual-basis LOTO scatter has no stable direction")
                    key = f"held/{held_task:03d}/target/{target:02d}/{side}"
                    basis_tensors[key] = eigenvectors[:, :retained].cpu().float()
                    energy = eigenvalues.sum().clamp_min(1e-30)
                    fits.append(
                        {
                            "held_task": held_task,
                            "target": target,
                            "family": target_metadata[target]["family"],
                            "side": side,
                            "native_width": int(eigenvectors.shape[0]),
                            "stable_rank": positive,
                            "stored_rank": retained,
                            "explained_energy": {
                                str(dimension): float(
                                    eigenvalues[: min(dimension, retained)].sum()
                                    / energy
                                )
                                for dimension in dimensions
                            },
                        }
                    )
                    del loto, eigenvalues, eigenvectors
                del scatters, total

    save_safetensors_atomic(basis_path, basis_tensors)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": BASIS_SCHEMA,
            "status": "complete",
            "question": (
                "whether fit-only minimum-norm native dual directions occupy a "
                "compact task-general target-specific subspace"
            ),
            "claim_boundary": (
                "task-balanced leave-one-task-out capacity of the tested rank-block "
                "basis dimensions only; non-pass does not reject all native bases, "
                "and pass does not prove Program-to-coefficient prediction or a "
                "G3 closed-loop Gate"
            ),
            "information_wall": {
                "roles": sorted(set(task_roles.values())),
                "held_authority_reads": 0,
                "action_meta_installed": False,
                "deployment_use": False,
                "task_indexed_loto_bases_are_evaluation_only": True,
                "deployment_requires_one_full_fit_refit_basis": True,
            },
            "fit": {
                "kind": "uncentered_task_equal_rank_block_projector_loto_pca",
                "precision": "float32",
                "output_basis": "shared_across_native_groups_within_each_target",
                "rank_gauge": (
                    "each video/member rank block contributes its normalized "
                    "row-space projector; invariant to nonsingular rank gauge"
                ),
                "basis_dimensions": list(dimensions),
                "score_bound": float(args.score_bound),
                "relative_singular_threshold": singular_threshold,
                "source_shard_count": int(args.shard_count),
                "task_count": len(tasks),
                "task_roles": {
                    role: sum(value == role for value in task_roles.values())
                    for role in sorted(set(task_roles.values()))
                },
                "source_record_count": source_records,
                "target_indices": list(target_indices),
                "dirty_exploratory_subset": bool(dirty_subset),
                "source_commit": source_commit,
                "scope": (
                    "four-family probe" if len(target_indices) == 4 else "target scan"
                ),
            },
            "tasks": [
                {"authority_id": task, "role": task_roles[task]} for task in tasks
            ],
            "selection": selection,
            "record_inventory": sorted(
                record_inventory,
                key=lambda row: (row["authority_id"], row["video_demo"]),
            ),
            "targets": [target_metadata[target] for target in target_indices],
            "native_geometry": {
                str(target): {
                    name: _quantiles(values)
                    for name, values in sorted(target_diagnostics[target].items())
                }
                for target in target_indices
            },
            "fits": fits,
            "basis_file": {
                "path": str(basis_path.resolve()),
                "bytes": basis_path.stat().st_size,
            },
            "git": aggregate_git,
        },
    )


def report(args: Any, *, repo_root: Path) -> None:
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise ValueError("dual-basis report output already exists")
    basis_manifest = read_json(args.output_dir / "basis_manifest.json")
    if (
        basis_manifest.get("schema_version") != BASIS_SCHEMA
        or basis_manifest.get("status") != "complete"
        or basis_manifest.get("fit", {}).get("source_shard_count")
        != args.shard_count
        or abs(
            float(basis_manifest.get("fit", {}).get("score_bound", -1.0))
            - args.score_bound
        )
        > 1e-12
    ):
        raise ValueError("dual-basis report lost its fitted basis authority")
    dimensions = tuple(map(int, basis_manifest["fit"]["basis_dimensions"]))
    basis_path = args.output_dir / "bases.safetensors"
    if (
        Path(str(basis_manifest["basis_file"]["path"])).resolve()
        != basis_path.resolve()
        or basis_manifest["basis_file"]["bytes"] != basis_path.stat().st_size
    ):
        raise ValueError("dual-basis report basis artifact changed")
    basis_authority = {
        "source_commit": basis_manifest["fit"]["source_commit"],
        "path": str(basis_path.resolve()),
        "bytes": basis_manifest["basis_file"]["bytes"],
    }
    formal_probe = not bool(basis_manifest["fit"]["dirty_exploratory_subset"])
    if formal_probe:
        basis_git = basis_manifest.get("git", {})
        if (
            basis_git.get("dirty_paths")
            or basis_git.get("authority_contains_commit") is not True
            or basis_git.get("commit") != basis_authority["source_commit"]
        ):
            raise ValueError("formal dual-basis aggregate authority changed")
        validate_formal_arguments(args, repo_root=repo_root)
        report_git = validate_formal_git(
            repo_root, expected_commit=basis_authority["source_commit"]
        )
    else:
        report_git = git_state(repo_root)
    worker_rows = []
    seen_tasks: set[int] = set()
    worker_summaries = []
    for shard in range(args.shard_count):
        path = args.output_dir / "replay_workers" / f"worker_{shard:03d}" / "results.json"
        payload = read_json(path)
        if (
            payload.get("schema_version") != REPLAY_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("shard_index") != shard
            or payload.get("shard_count") != args.shard_count
            or abs(float(payload.get("score_bound")) - args.score_bound) > 1e-12
            or payload.get("basis_authority") != basis_authority
            or payload.get("row_count") != len(payload.get("rows", ()))
        ):
            raise ValueError("dual-basis replay worker contract changed")
        replay_git = payload.get("git", {})
        if formal_probe and (
            replay_git.get("dirty_paths")
            or replay_git.get("authority_contains_commit") is not True
            or replay_git.get("commit") != basis_authority["source_commit"]
        ):
            raise ValueError("formal dual-basis replay was not clean frozen authority")
        tasks = set(map(int, payload["task_ids"]))
        if {int(row["authority_id"]) for row in payload["rows"]} != tasks:
            raise ValueError("dual-basis replay worker row inventory changed")
        if seen_tasks & tasks:
            raise ValueError("dual-basis replay duplicated tasks")
        seen_tasks.update(tasks)
        worker_rows.extend(payload["rows"])
        worker_summaries.append(
            {
                "shard_index": shard,
                "task_count": len(tasks),
                "row_count": payload["row_count"],
                "max_cuda_allocated_bytes": payload["max_cuda_allocated_bytes"],
            }
        )
    expected_tasks = {
        int(row["authority_id"]) for row in basis_manifest["tasks"]
    }
    if seen_tasks != expected_tasks:
        raise ValueError("dual-basis replay lost LOTO task coverage")

    target_rows = {
        int(row["target"]): row for row in basis_manifest["targets"]
    }
    task_roles = {
        int(row["authority_id"]): str(row["role"])
        for row in basis_manifest["tasks"]
    }
    expected_labels: tuple[str | int, ...] = ("full", *dimensions)
    expected_rows = {
        (
            int(record["authority_id"]),
            int(record["video_demo"]),
            str(member),
            target,
            label,
        )
        for record in basis_manifest["record_inventory"]
        for member in record["member_names"]
        for target in target_rows
        for label in expected_labels
    }
    actual_rows = Counter(
        (
            int(row["authority_id"]),
            int(row["video_demo"]),
            str(row["member_name"]),
            int(row["target"]),
            row["dimension"],
        )
        for row in worker_rows
    )
    if set(actual_rows) != expected_rows or any(
        count != 1 for count in actual_rows.values()
    ):
        raise ValueError("dual-basis replay lost or duplicated an expected row")
    if any(
        str(row["role"]) != task_roles[int(row["authority_id"])]
        or str(row["family"])
        != str(target_rows[int(row["target"])]["family"])
        for row in worker_rows
    ):
        raise ValueError("dual-basis replay row authority changed")

    exact_key = lambda row: (
        int(row["authority_id"]),
        int(row["video_demo"]),
        str(row["member_name"]),
        int(row["target"]),
    )
    full_by_key = {
        exact_key(row): float(row["update_cosine"])
        for row in worker_rows
        if row["dimension"] == "full"
    }
    labels = expected_labels
    curves = []
    for label in labels:
        selected = [row for row in worker_rows if row["dimension"] == label]
        if not selected:
            raise ValueError("dual-basis report lost a curve dimension")
        input_factors = []
        output_factors = []
        input_subspaces = []
        output_subspaces = []
        spectrum_errors = []
        minimum_singular_ratios = []
        ratio_rows = []
        for row in selected:
            input_factors.append(float(row["input_factor_cosine_mean"]))
            output_factors.append(float(row["output_factor_cosine_mean"]))
            input_subspaces.append(float(row["input_subspace_similarity"]))
            output_subspaces.append(float(row["output_subspace_similarity"]))
            spectrum_errors.append(float(row["small_core_log_spectrum_rmse"]))
            minimum_singular_ratios.append(
                float(row["minimum_small_core_singular_to_teacher"])
            )
            reference = full_by_key[exact_key(row)]
            enriched = dict(row)
            enriched["projected_to_full_update_cosine_ratio"] = (
                float(row["update_cosine"]) / max(reference, 1e-12)
            )
            enriched["projected_minus_full_update_cosine"] = (
                float(row["update_cosine"]) - reference
            )
            ratio_rows.append(enriched)
        overall = _task_distribution(selected, "update_cosine")
        families = {str(row["family"]) for row in selected}
        if formal_probe and families != {"q", "v", "action_in", "action_out"}:
            raise ValueError("formal dual-basis report lost a target family")
        family_rows = {}
        for family in sorted(families):
            family_selected = [
                row for row in selected if str(row["family"]) == family
            ]
            family_tasks = {int(row["authority_id"]) for row in family_selected}
            if family_tasks != expected_tasks:
                raise ValueError("dual-basis family lost task coverage")
            family_rows[family] = _task_distribution(
                family_selected, "update_cosine"
            )
        family_pass = {
            family: all(_probe_thresholds(values).values())
            for family, values in family_rows.items()
        }
        overall_thresholds = _probe_thresholds(overall)
        effective_dimensions = {
            str(target): {
                "input": sorted(
                    {
                        int(row["effective_input_dimension"])
                        for row in selected
                        if int(row["target"]) == target
                    }
                ),
                "output": sorted(
                    {
                        int(row["effective_output_dimension"])
                        for row in selected
                        if int(row["target"]) == target
                    }
                ),
            }
            for target in sorted({int(row["target"]) for row in selected})
        }
        curves.append(
            {
                "dimension": label,
                "effective_dimensions": effective_dimensions,
                "overall_task_equal_update_cosine": overall,
                "families_task_equal_update_cosine": family_rows,
                "projected_to_full_update_cosine_ratio": _task_distribution(
                    ratio_rows, "projected_to_full_update_cosine_ratio"
                ),
                "projected_minus_full_update_cosine": _task_distribution(
                    ratio_rows, "projected_minus_full_update_cosine"
                ),
                "input_factor_cosine": _quantiles(input_factors),
                "output_factor_cosine": _quantiles(output_factors),
                "input_subspace_similarity": _quantiles(input_subspaces),
                "output_subspace_similarity": _quantiles(output_subspaces),
                "small_core_log_spectrum_rmse": _quantiles(spectrum_errors),
                "minimum_small_core_singular_to_teacher": _quantiles(
                    minimum_singular_ratios
                ),
                "probe_thresholds": {
                    **overall_thresholds,
                    "all_four_families_pass": families
                    == {"q", "v", "action_in", "action_out"}
                    and all(family_pass.values()),
                    "family_pass": family_pass,
                },
            }
        )
        curves[-1]["probe_thresholds"]["raw_pass"] = all(
            value
            for key, value in curves[-1]["probe_thresholds"].items()
            if key != "family_pass"
        )
    full_reference_valid = bool(curves[0]["probe_thresholds"]["raw_pass"])
    for curve in curves:
        curve["probe_thresholds"]["pass"] = bool(
            full_reference_valid and curve["probe_thresholds"]["raw_pass"]
        )
    passing = [
        row["dimension"]
        for row in curves
        if formal_probe
        and row["dimension"] != "full"
        and row["probe_thresholds"]["pass"]
    ]
    write_json_atomic(
        report_path,
        {
            "schema_version": REPORT_SCHEMA,
            "status": "complete",
            "question": basis_manifest["question"],
            "claim_boundary": basis_manifest["claim_boundary"],
            "pooling": {
                "kind": "query_key_antithetic_signed_cross_attention_replay",
                "score_bound": args.score_bound,
                "full_dual_is_code_reference_not_a_deployment_path": True,
                "bank_max_score_calibration_is_oracle_only": True,
                "output_gain_is_shared_across_all_native_groups_per_target_rank": True,
            },
            "gate_contract": {
                "overall_task_equal_update_cosine_median": 0.98,
                "overall_task_equal_update_cosine_p10": 0.95,
                "each_family_task_equal_median": 0.98,
                "each_family_task_equal_p10": 0.95,
                "at_least_90_percent_tasks_at_update_cosine_0.95": True,
                "same_thresholds_apply_to_task_worst_video": True,
                "selection_rule": "smallest preregistered basis dimension passing all",
                "scope": "four-family probe; expand all 38 targets before compiler use",
            },
            "curves": curves,
            "full_dual_reference_valid": full_reference_valid,
            "selected_probe_dimension": min(passing) if passing else None,
            "probe_gate_pass": bool(formal_probe and passing),
            "formal_contract": formal_probe,
            "worker_summaries": worker_summaries,
            "fit": basis_manifest["fit"],
            "information_wall": basis_manifest["information_wall"],
            "git": report_git,
        },
    )
