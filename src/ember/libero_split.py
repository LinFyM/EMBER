"""Pure deterministic split search and role-coverage audit."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SEARCH_ALGORITHM = "sha256_multistart_greedy_plus_steepest_swap_v1"
SEARCH_SEED = 20_260_718
SEARCH_CANDIDATE_COUNT = 16_384
MINIMUM_SOURCE_ROLE_OCCURRENCES = 2
CONTENT_ROLE_PREFIXES = (
    "moved_object:",
    "target_receptacle:",
    "target_relation:",
    "source_selector:",
    "target_selector:",
    "actuated_fixture:",
    "actuated_selector:",
)


class SplitResealError(ValueError):
    """Raised when role coverage or split constraints cannot be sealed."""


def _sorted_split(split: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    return {
        name: sorted(int(index) for index in split[name])
        for name in ("source", "validation", "held_out")
    }


def _task_map(tasks: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    by_index = {int(task["task_index"]): task for task in tasks}
    if len(by_index) != len(tasks):
        raise SplitResealError("task factor table contains duplicate task indices")
    return by_index


def _split_membership(
    tasks: Sequence[Mapping[str, Any]],
    source: Sequence[int],
    validation: Sequence[int],
    held_out: Sequence[int],
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    split = _sorted_split(
        {"source": source, "validation": validation, "held_out": held_out}
    )
    expected = set(_task_map(tasks))
    selected = [index for name in split for index in split[name]]
    violations: list[dict[str, Any]] = []
    if len(selected) != len(set(selected)):
        violations.append({"code": "split_overlap"})
    if set(selected) != expected:
        violations.append(
            {
                "code": "split_coverage",
                "missing": sorted(expected - set(selected)),
                "extra": sorted(set(selected) - expected),
            }
        )
    return split, violations


def _content_atoms(task: Mapping[str, Any]) -> set[str]:
    return {
        atom
        for atom in task["primitive_role_atoms"]
        if str(atom).startswith(CONTENT_ROLE_PREFIXES)
    }


def audit_split(
    tasks: Sequence[Mapping[str, Any]],
    *,
    source: Sequence[int],
    validation: Sequence[int],
    held_out: Sequence[int],
    minimum: int = MINIMUM_SOURCE_ROLE_OCCURRENCES,
) -> dict[str, Any]:
    if not isinstance(minimum, int) or minimum < 1:
        raise SplitResealError("minimum source-role occurrence threshold must be positive")
    by_index = _task_map(tasks)
    split, structural_violations = _split_membership(tasks, source, validation, held_out)
    source_set = set(split["source"])
    evaluation_set = set(split["validation"] + split["held_out"])

    source_atoms: defaultdict[str, list[int]] = defaultdict(list)
    evaluation_atoms: defaultdict[str, list[int]] = defaultdict(list)
    for index, task in by_index.items():
        destination = source_atoms if index in source_set else evaluation_atoms
        for atom in task["primitive_role_atoms"]:
            destination[str(atom)].append(index)
    coverage_violations = [
        {
            "atom": atom,
            "source_count": len(source_atoms.get(atom, [])),
            "source_task_indices": sorted(source_atoms.get(atom, [])),
            "evaluation_count": len(indices),
            "evaluation_task_indices": sorted(indices),
            "required_source_count": minimum,
        }
        for atom, indices in sorted(evaluation_atoms.items())
        if len(source_atoms.get(atom, [])) < minimum
    ]

    source_compositions = {
        str(by_index[index]["composition_signature"]) for index in source_set
    }
    same_scene_source = []
    same_scene_hard_negative = []
    novel_compositions = []
    for index in sorted(evaluation_set):
        task = by_index[index]
        peers = [
            source_index
            for source_index in sorted(source_set)
            if by_index[source_index]["scene"] == task["scene"]
            and by_index[source_index]["composition_signature"]
            != task["composition_signature"]
        ]
        hard_peers = [
            source_index
            for source_index in peers
            if _content_atoms(by_index[source_index]).intersection(_content_atoms(task))
        ]
        if peers:
            same_scene_source.append(index)
        if hard_peers:
            same_scene_hard_negative.append(index)
        if str(task["composition_signature"]) not in source_compositions:
            novel_compositions.append(index)

    scenes = sorted({str(task["scene"]) for task in tasks})
    difficulties = sorted(
        {int(task["difficulty"]["operation_count"]) for task in tasks}
    )
    scene_counts = {
        name: {
            scene: sum(by_index[index]["scene"] == scene for index in split[name])
            for scene in scenes
        }
        for name in split
    }
    difficulty_counts = {
        name: {
            str(level): sum(
                int(by_index[index]["difficulty"]["operation_count"]) == level
                for index in split[name]
            )
            for level in difficulties
        }
        for name in split
    }
    all_source_counts = sorted(
        len(source_atoms.get(atom, [])) for atom in evaluation_atoms
    )
    return {
        "mechanics_valid": not structural_violations and not coverage_violations,
        "minimum_source_role_occurrences": minimum,
        "structural_violations": structural_violations,
        "coverage_violations": coverage_violations,
        "evaluation_role_count": len(evaluation_atoms),
        "minimum_observed_source_count_for_evaluation_roles": (
            all_source_counts[0] if all_source_counts else None
        ),
        "novel_full_composition_task_indices": novel_compositions,
        "novel_full_composition_count": len(novel_compositions),
        "same_scene_source_task_indices": same_scene_source,
        "same_scene_source_count": len(same_scene_source),
        "same_scene_hard_negative_task_indices": same_scene_hard_negative,
        "same_scene_hard_negative_count": len(same_scene_hard_negative),
        "scene_counts": scene_counts,
        "difficulty_counts": difficulty_counts,
    }


def _hash_rank(seed: int, restart: int, label: str, index: int) -> bytes:
    return hashlib.sha256(f"{seed}:{restart}:{label}:{index}".encode("ascii")).digest()


def _evaluation_feasible(
    evaluation: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    atom_totals: Mapping[str, int],
    minimum: int,
) -> bool:
    counts: Counter[str] = Counter(
        atom for index in evaluation for atom in by_index[index]["primitive_role_atoms"]
    )
    return all(
        atom_totals[atom] - count >= minimum for atom, count in counts.items()
    )


def _distribution_penalty(
    selected: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    field: str,
    nested: str | None = None,
) -> int:
    if nested is None:
        value = lambda task: task[field]
    else:
        value = lambda task: task[field][nested]
    totals = Counter(value(task) for task in by_index.values())
    selected_counts = Counter(value(by_index[index]) for index in selected)
    total_size = len(by_index)
    selected_size = len(selected)
    return sum(
        (total_size * selected_counts[key] - selected_size * total) ** 2
        for key, total in totals.items()
    )


def _combined_metrics(
    evaluation: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    prior_evaluation: set[int],
) -> dict[str, Any]:
    source = set(by_index) - evaluation
    source_compositions = {by_index[index]["composition_signature"] for index in source}
    novel = sum(
        by_index[index]["composition_signature"] not in source_compositions
        for index in evaluation
    )
    hard_negative = 0
    scene_source = 0
    for index in evaluation:
        task = by_index[index]
        peers = [
            source_index
            for source_index in source
            if by_index[source_index]["scene"] == task["scene"]
            and by_index[source_index]["composition_signature"]
            != task["composition_signature"]
        ]
        scene_source += bool(peers)
        hard_negative += any(
            _content_atoms(by_index[source_index]).intersection(_content_atoms(task))
            for source_index in peers
        )
    return {
        "novel_full_composition_count": novel,
        "same_scene_hard_negative_count": hard_negative,
        "same_scene_source_count": scene_source,
        "scene_distribution_penalty": _distribution_penalty(
            evaluation, by_index=by_index, field="scene"
        ),
        "difficulty_distribution_penalty": _distribution_penalty(
            evaluation,
            by_index=by_index,
            field="difficulty",
            nested="operation_count",
        ),
        "prior_evaluation_retained_count": len(evaluation.intersection(prior_evaluation)),
    }


def _combined_score(metrics: Mapping[str, Any], evaluation: set[int]) -> tuple[Any, ...]:
    return (
        metrics["novel_full_composition_count"],
        metrics["same_scene_hard_negative_count"],
        metrics["same_scene_source_count"],
        -metrics["scene_distribution_penalty"],
        -metrics["difficulty_distribution_penalty"],
        metrics["prior_evaluation_retained_count"],
        tuple(-index for index in sorted(evaluation)),
    )


def _partition_metrics(
    validation: set[int],
    evaluation: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    prior_validation: set[int],
    prior_held: set[int],
) -> dict[str, Any]:
    held = evaluation - validation
    scenes = {by_index[index]["scene"] for index in evaluation}
    shared_scenes = sum(
        any(by_index[index]["scene"] == scene for index in validation)
        and any(by_index[index]["scene"] == scene for index in held)
        for scene in scenes
    )
    scene_imbalance = sum(
        (
            sum(by_index[index]["scene"] == scene for index in validation)
            - sum(by_index[index]["scene"] == scene for index in held)
        )
        ** 2
        for scene in scenes
    )
    levels = {
        int(by_index[index]["difficulty"]["operation_count"]) for index in evaluation
    }
    difficulty_imbalance = sum(
        (
            sum(
                int(by_index[index]["difficulty"]["operation_count"]) == level
                for index in validation
            )
            - sum(
                int(by_index[index]["difficulty"]["operation_count"]) == level
                for index in held
            )
        )
        ** 2
        for level in levels
    )
    validation_atoms = Counter(
        atom for index in validation for atom in by_index[index]["primitive_role_atoms"]
    )
    held_atoms = Counter(
        atom for index in held for atom in by_index[index]["primitive_role_atoms"]
    )
    factor_imbalance = sum(
        abs(validation_atoms[atom] - held_atoms[atom])
        for atom in set(validation_atoms).union(held_atoms)
    )
    role_retained = len(validation.intersection(prior_validation)) + len(
        held.intersection(prior_held)
    )
    return {
        "shared_evaluation_scene_count": shared_scenes,
        "scene_imbalance_penalty": scene_imbalance,
        "difficulty_imbalance_penalty": difficulty_imbalance,
        "primitive_role_imbalance_penalty": factor_imbalance,
        "prior_exact_role_retained_count": role_retained,
    }


def _partition_score(metrics: Mapping[str, Any], validation: set[int]) -> tuple[Any, ...]:
    return (
        metrics["shared_evaluation_scene_count"],
        -metrics["scene_imbalance_penalty"],
        -metrics["difficulty_imbalance_penalty"],
        -metrics["primitive_role_imbalance_penalty"],
        metrics["prior_exact_role_retained_count"],
        tuple(-index for index in sorted(validation)),
    )


def _steepest_combined_swap(
    evaluation: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    atom_totals: Mapping[str, int],
    minimum: int,
    prior_evaluation: set[int],
) -> set[int]:
    current = set(evaluation)
    while True:
        current_metrics = _combined_metrics(
            current, by_index=by_index, prior_evaluation=prior_evaluation
        )
        current_score = _combined_score(current_metrics, current)
        best = current
        best_score = current_score
        for removed in sorted(current):
            for added in sorted(set(by_index) - current):
                candidate = (current - {removed}) | {added}
                if not _evaluation_feasible(
                    candidate,
                    by_index=by_index,
                    atom_totals=atom_totals,
                    minimum=minimum,
                ):
                    continue
                metrics = _combined_metrics(
                    candidate, by_index=by_index, prior_evaluation=prior_evaluation
                )
                score = _combined_score(metrics, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score
        if best == current:
            return current
        current = best


def _steepest_partition_swap(
    validation: set[int],
    evaluation: set[int],
    *,
    by_index: Mapping[int, Mapping[str, Any]],
    prior_validation: set[int],
    prior_held: set[int],
) -> set[int]:
    current = set(validation)
    while True:
        metrics = _partition_metrics(
            current,
            evaluation,
            by_index=by_index,
            prior_validation=prior_validation,
            prior_held=prior_held,
        )
        current_score = _partition_score(metrics, current)
        best = current
        best_score = current_score
        for removed in sorted(current):
            for added in sorted(evaluation - current):
                candidate = (current - {removed}) | {added}
                candidate_metrics = _partition_metrics(
                    candidate,
                    evaluation,
                    by_index=by_index,
                    prior_validation=prior_validation,
                    prior_held=prior_held,
                )
                score = _partition_score(candidate_metrics, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score
        if best == current:
            return current
        current = best


def search_split(
    tasks: Sequence[Mapping[str, Any]],
    *,
    prior_split: Mapping[str, Sequence[int]],
    sizes: Mapping[str, int],
    minimum: int = MINIMUM_SOURCE_ROLE_OCCURRENCES,
    seed: int = SEARCH_SEED,
    candidate_count: int = SEARCH_CANDIDATE_COUNT,
) -> dict[str, Any]:
    by_index = _task_map(tasks)
    if sum(sizes.values()) != len(tasks) or set(sizes) != {
        "source",
        "validation",
        "held_out",
    }:
        raise SplitResealError("split sizes must cover the full task table")
    if sizes["validation"] + sizes["held_out"] <= 0:
        raise SplitResealError("split search requires a non-empty evaluation surface")
    if candidate_count < 1:
        raise SplitResealError("candidate count must be positive")
    prior = _sorted_split(prior_split)
    prior_evaluation = set(prior["validation"] + prior["held_out"])
    atom_totals = Counter(
        atom for task in tasks for atom in task["primitive_role_atoms"]
    )
    forced_source = {
        int(task["task_index"])
        for task in tasks
        if any(atom_totals[atom] <= minimum for atom in task["primitive_role_atoms"])
    }
    if len(forced_source) > sizes["source"]:
        raise SplitResealError(
            "split search infeasible: role coverage forces "
            f"{len(forced_source)} source tasks into {sizes['source']} slots"
        )

    evaluation_size = sizes["validation"] + sizes["held_out"]
    best_evaluation: set[int] | None = None
    best_score: tuple[Any, ...] | None = None
    for restart in range(candidate_count):
        order = sorted(
            by_index,
            key=lambda index: (_hash_rank(seed, restart, "combined", index), index),
        )
        evaluation: set[int] = set()
        for index in order:
            if len(evaluation) == evaluation_size:
                break
            candidate = evaluation | {index}
            if _evaluation_feasible(
                candidate,
                by_index=by_index,
                atom_totals=atom_totals,
                minimum=minimum,
            ):
                evaluation = candidate
        if len(evaluation) != evaluation_size:
            continue
        metrics = _combined_metrics(
            evaluation, by_index=by_index, prior_evaluation=prior_evaluation
        )
        score = _combined_score(metrics, evaluation)
        if best_score is None or score > best_score:
            best_evaluation = evaluation
            best_score = score
    if best_evaluation is None:
        raise SplitResealError(
            "split search infeasible: deterministic candidates could not fill the evaluation surface"
        )

    evaluation = _steepest_combined_swap(
        best_evaluation,
        by_index=by_index,
        atom_totals=atom_totals,
        minimum=minimum,
        prior_evaluation=prior_evaluation,
    )
    best_validation: set[int] | None = None
    best_partition_score: tuple[Any, ...] | None = None
    for restart in range(candidate_count):
        order = sorted(
            evaluation,
            key=lambda index: (_hash_rank(seed, restart, "partition", index), index),
        )
        validation = set(order[: sizes["validation"]])
        metrics = _partition_metrics(
            validation,
            evaluation,
            by_index=by_index,
            prior_validation=set(prior["validation"]),
            prior_held=set(prior["held_out"]),
        )
        score = _partition_score(metrics, validation)
        if best_partition_score is None or score > best_partition_score:
            best_validation = validation
            best_partition_score = score
    if best_validation is None:
        raise SplitResealError("split search infeasible: validation/held partition failed")
    validation = _steepest_partition_swap(
        best_validation,
        evaluation,
        by_index=by_index,
        prior_validation=set(prior["validation"]),
        prior_held=set(prior["held_out"]),
    )
    split = {
        "source": sorted(set(by_index) - evaluation),
        "validation": sorted(validation),
        "held_out": sorted(evaluation - validation),
    }
    audit = audit_split(tasks, **split, minimum=minimum)
    if not audit["mechanics_valid"]:
        raise SplitResealError("split search produced an invalid result")
    combined_metrics = _combined_metrics(
        evaluation, by_index=by_index, prior_evaluation=prior_evaluation
    )
    partition_metrics = _partition_metrics(
        validation,
        evaluation,
        by_index=by_index,
        prior_validation=set(prior["validation"]),
        prior_held=set(prior["held_out"]),
    )
    return {
        "algorithm": SEARCH_ALGORITHM,
        "seed": seed,
        "candidate_count": candidate_count,
        "split": split,
        "combined_objective": combined_metrics,
        "partition_objective": partition_metrics,
        "audit": audit,
    }
