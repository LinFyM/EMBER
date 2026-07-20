"""Specification-only construction and audit of the EMBER LIBERO-90 split."""

from __future__ import annotations

import importlib.metadata
import inspect
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from ember.libero_data import sha256_file
from ember.libero_task_factors import (
    FACTOR_ROLE_DEFINITIONS,
    FACTOR_SCHEMA,
    factor_task,
)


SCHEMA_VERSION = "ember_libero90_protocol_v1"
SPLIT_SEED = 20260720
SCENE_RE = re.compile(r"^([A-Z]+(?:_[A-Z]+)*_SCENE\d+)_")
SPLIT_ORDER = ("train", "validation", "test")
HELD_SPLITS = ("validation", "test")
SCENE_FAMILY_QUOTAS = {"KITCHEN": 5, "LIVING_ROOM": 3, "STUDY": 2}
DIFFICULTY_QUOTAS = {1: 5, 2: 5}
TASK_FAMILY_QUOTAS = {
    "actuation": 2,
    "single_place": 3,
    "pick_place": 4,
    "compound": 1,
    "stack": 0,
}

Constraint = tuple[dict[int, float], float, float]


class ProtocolError(RuntimeError):
    """Raised when the pinned protocol cannot be reproduced exactly."""


def _task_family(row: dict[str, Any]) -> str:
    signature = row["order_signature"]
    if signature == "place":
        return "single_place"
    if signature == "pick_up>place":
        return "pick_place"
    if signature.startswith("stack"):
        return "stack"
    if len(row["steps"]) == 1 and not row["steps"][0]["moved_objects"]:
        return "actuation"
    return "compound"


def load_specification_surface() -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Read only pinned task identity, scene, and language from LIBERO-90."""

    from libero.libero import benchmark
    from libero.libero.benchmark import libero_suite_task_map

    version = importlib.metadata.version("hf-libero")
    if version != "0.1.4":
        raise ProtocolError("the protocol requires hf-libero==0.1.4")
    suite = benchmark.get_benchmark("libero_90")()
    if suite.get_num_tasks() != 90:
        raise ProtocolError("the pinned LIBERO-90 suite does not contain 90 tasks")
    rows = []
    for task_index in range(90):
        task = suite.get_task(task_index)
        matched = SCENE_RE.match(task.name)
        if matched is None:
            raise ProtocolError(f"unrecognized LIBERO-90 task name: {task.name}")
        row = factor_task(task_index=task_index, scene=matched.group(1), language=task.language)
        row.update(
            {
                "task_name": task.name,
                "bddl_basename": task.bddl_file,
                "init_states_basename": task.init_states_file,
                "scene_family": matched.group(1).rsplit("_SCENE", 1)[0],
                "task_family": _task_family(row),
            }
        )
        rows.append(row)
    return suite, rows, {
        "distribution": "hf-libero",
        "version": version,
        "benchmark_module_sha256": sha256_file(Path(inspect.getfile(benchmark))),
        "task_map_module_sha256": sha256_file(Path(inspect.getfile(libero_suite_task_map))),
        "factor_parser_sha256": sha256_file(Path(inspect.getfile(factor_task))),
    }


def build_factor_artifact(
    rows: list[dict[str, Any]], authority: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "factor_schema": FACTOR_SCHEMA,
        "identity_fields_not_used_as_factors": ["task_index", "task_name"],
        "split_feature_fields": ["scene", "language", "derived_role_factors"],
        "forbidden_input_fields": [
            "action",
            "proprio",
            "reward",
            "terminal",
            "normalization",
            "policy_outcome",
        ],
        "role_definitions": FACTOR_ROLE_DEFINITIONS,
        "specification_authority": authority,
        "rows": rows,
    }


def _add(constraints: list[Constraint], coefficients: dict[int, float], lo: float, hi: float) -> None:
    constraints.append((coefficients, lo, hi))


def _members(
    rows: list[dict[str, Any]], offset: int, predicate: Callable[[dict[str, Any]], bool]
) -> dict[int, float]:
    return {offset + row["task_index"]: 1 for row in rows if predicate(row)}


def _add_held_distribution_constraints(rows: list[dict[str, Any]], constraints: list[Constraint]) -> None:
    task_count = len(rows)
    for split_offset in range(len(HELD_SPLITS)):
        offset = split_offset * task_count
        _add(constraints, {offset + index: 1 for index in range(task_count)}, 10, 10)
        for family, count in SCENE_FAMILY_QUOTAS.items():
            _add(constraints, _members(rows, offset, lambda row, f=family: row["scene_family"] == f), count, count)
        for difficulty, count in DIFFICULTY_QUOTAS.items():
            _add(
                constraints,
                _members(
                    rows,
                    offset,
                    lambda row, d=difficulty: row["difficulty"]["operation_count"] == d,
                ),
                count,
                count,
            )
        for family, count in TASK_FAMILY_QUOTAS.items():
            _add(constraints, _members(rows, offset, lambda row, f=family: row["task_family"] == f), count, count)
        for scene in sorted({row["scene"] for row in rows}):
            _add(constraints, _members(rows, offset, lambda row, s=scene: row["scene"] == s), 0, 1)


def _add_role_support_constraints(rows: list[dict[str, Any]], constraints: list[Constraint]) -> None:
    task_count = len(rows)
    atoms: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for atom in row["primitive_role_atoms"]:
            atoms[atom].append(row["task_index"])
    for members in atoms.values():
        coefficients = {index: 1 for index in members}
        coefficients.update({task_count + index: 1 for index in members})
        _add(constraints, coefficients, 0, max(len(members) - 2, 0))


def _add_composition_constraints(rows: list[dict[str, Any]], constraints: list[Constraint]) -> None:
    task_count = len(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["composition_signature"]].append(row["task_index"])
    for members in groups.values():
        for split_offset in range(len(HELD_SPLITS)):
            offset = split_offset * task_count
            for task_index in members[1:]:
                _add(constraints, {offset + members[0]: 1, offset + task_index: -1}, 0, 0)
        if len(members) > 1 and rows[members[0]]["task_family"] != "actuation":
            for split_offset in range(len(HELD_SPLITS)):
                offset = split_offset * task_count
                _add(constraints, {offset + task_index: 1 for task_index in members}, 0, 0)


def _feature_memberships(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        task = row["task_index"]
        result[f"scene:{row['scene']}"].append(task)
        result[f"order:{row['order_signature']}"].append(task)
        for atom in row["primitive_role_atoms"]:
            result[f"role:{atom}"].append(task)
    return dict(sorted(result.items()))


def _difference_constraints(
    constraints: list[Constraint], aux: int, left: dict[int, float], right: dict[int, float]
) -> None:
    _add(constraints, {aux: 1, **{k: -v for k, v in left.items()}, **right}, 0, math.inf)
    _add(constraints, {aux: 1, **left, **{k: -v for k, v in right.items()}}, 0, math.inf)


def _add_feature_objectives(
    rows: list[dict[str, Any]], constraints: list[Constraint]
) -> tuple[np.ndarray, int]:
    task_count = len(rows)
    memberships = _feature_memberships(rows)
    variable_count = 2 * task_count + 3 * len(memberships)
    objectives = np.zeros((3, variable_count))
    next_aux = 2 * task_count
    for members in memberships.values():
        total = len(members)
        validation = {index: 1 for index in members}
        test = {task_count + index: 1 for index in members}
        _difference_constraints(constraints, next_aux, validation, test)
        for aux, split_members in ((next_aux + 1, validation), (next_aux + 2, test)):
            _add(constraints, {aux: 1, **{k: -9 * v for k, v in split_members.items()}}, -total, math.inf)
            _add(constraints, {aux: 1, **{k: 9 * v for k, v in split_members.items()}}, total, math.inf)
        objectives[0, next_aux] = max(1, 1000 // total)
        objectives[1, next_aux + 1 : next_aux + 3] = max(1, 100 // total)
        next_aux += 3
    for split_offset, split_name in enumerate(HELD_SPLITS):
        for row in rows:
            payload = (
                f"{SPLIT_SEED}\0{split_name}\0{row['scene']}\0{row['language']}"
            ).encode("utf-8")
            weight = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000 + 1
            objectives[2, split_offset * task_count + row["task_index"]] = weight
    return objectives, len(memberships)


def _build_problem(rows: list[dict[str, Any]]) -> tuple[list[Constraint], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    constraints: list[Constraint] = []
    _add_held_distribution_constraints(rows, constraints)
    task_count = len(rows)
    for task in range(task_count):
        _add(constraints, {task: 1, task_count + task: 1}, 0, 1)
    _add_role_support_constraints(rows, constraints)
    _add_composition_constraints(rows, constraints)
    objectives, feature_count = _add_feature_objectives(rows, constraints)
    variable_count = objectives.shape[1]
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[: 2 * task_count] = 1
    integrality = np.zeros(variable_count)
    integrality[: 2 * task_count] = 1
    return constraints, lower, upper, integrality, objectives, feature_count


def _linear_constraint(constraints: list[Constraint], variable_count: int) -> LinearConstraint:
    matrix = lil_matrix((len(constraints), variable_count), dtype=np.float64)
    lower = np.empty(len(constraints))
    upper = np.empty(len(constraints))
    for row_index, (coefficients, lo, hi) in enumerate(constraints):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index] = lo
        upper[row_index] = hi
    return LinearConstraint(matrix.tocsr(), lower, upper)


def _solve_stage(
    constraints: list[Constraint], lower: np.ndarray, upper: np.ndarray, integrality: np.ndarray, objective: np.ndarray
) -> Any:
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=_linear_constraint(constraints, len(lower)),
        options={"mip_rel_gap": 0.0, "presolve": True, "time_limit": 120},
    )
    if not result.success or result.x is None:
        raise ProtocolError(f"split MILP did not reach an optimum: {result.message}")
    return result


def seal_split(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    constraints, lower, upper, integrality, objectives, feature_count = _build_problem(rows)
    base_constraints = len(constraints)
    optima = []
    result = None
    for stage, objective in enumerate(objectives):
        result = _solve_stage(constraints, lower, upper, integrality, objective)
        value = float(np.dot(objective, result.x))
        optimum = int(round(value))
        if not np.isclose(value, optimum, atol=1e-5):
            raise ProtocolError(f"non-integral objective at lexicographic stage {stage + 1}")
        optima.append(optimum)
        if stage < len(objectives) - 1:
            _add(constraints, {i: float(v) for i, v in enumerate(objective) if v}, optimum, optimum)
    assert result is not None
    validation = [index for index in range(90) if result.x[index] > 0.5]
    test = [index for index in range(90) if result.x[90 + index] > 0.5]
    held = set(validation) | set(test)
    split = {"train": [i for i in range(90) if i not in held], "validation": validation, "test": test}
    metadata = {
        "algorithm": "scipy_milp_highs_three_stage_lexicographic_v1",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "seed": SPLIT_SEED,
        "tie_break": "sha256(seed,split,scene,language) modulo 1000000 plus 1",
        "binary_variables": 180,
        "auxiliary_variables": len(lower) - 180,
        "feature_count": feature_count,
        "base_constraint_count": base_constraints,
        "final_constraint_count": len(constraints),
        "stage_optima": dict(
            zip(
                ("validation_test_factor_imbalance", "global_marginal_deviation", "seeded_tie_break"),
                optima,
                strict=True,
            )
        ),
    }
    return split, metadata


def _summary(rows: list[dict[str, Any]], task_ids: list[int]) -> dict[str, Any]:
    selected = [rows[index] for index in task_ids]
    return {
        "count": len(selected),
        "scene_family": dict(sorted(Counter(row["scene_family"] for row in selected).items())),
        "scene": dict(sorted(Counter(row["scene"] for row in selected).items())),
        "task_family": dict(sorted(Counter(row["task_family"] for row in selected).items())),
        "operation_count": {
            str(key): value
            for key, value in sorted(Counter(row["difficulty"]["operation_count"] for row in selected).items())
        },
        "order_signature": dict(sorted(Counter(row["order_signature"] for row in selected).items())),
    }


def _composition_owners(
    rows: list[dict[str, Any]], split_by_id: dict[int, str]
) -> dict[str, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["composition_signature"]].append(row["task_index"])
    owners = {}
    for signature, members in groups.items():
        group_owners = {split_by_id[task] for task in members}
        if len(group_owners) != 1:
            raise ProtocolError("an exact composition crosses split boundaries")
        owners[signature] = next(iter(group_owners))
    return owners


def _held_support_and_neighbors(
    rows: list[dict[str, Any]], split: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = set(split["train"])
    train_counts = Counter(atom for task in train for atom in rows[task]["primitive_role_atoms"])
    support = []
    neighbors = []
    for name in HELD_SPLITS:
        for task in split[name]:
            row = rows[task]
            counts = {atom: train_counts[atom] for atom in row["primitive_role_atoms"]}
            if min(counts.values()) < 2:
                raise ProtocolError(f"held task {task} lacks multiple train supports for a role")
            support.append({"split": name, "task_index": task, "train_counts": counts})
            held_atoms = set(row["primitive_role_atoms"])
            candidates = []
            for train_task in train:
                if rows[train_task]["task_family"] != row["task_family"]:
                    continue
                other = set(rows[train_task]["primitive_role_atoms"])
                candidates.append((len(held_atoms & other) / len(held_atoms | other), train_task, len(held_atoms & other), len(held_atoms | other)))
            candidates.sort(key=lambda value: (-value[0], value[1]))
            neighbors.append(
                {
                    "split": name,
                    "task_index": task,
                    "same_family_train_neighbors": [
                        {"task_index": item[1], "jaccard_numerator": item[2], "jaccard_denominator": item[3]}
                        for item in candidates[:3]
                    ],
                }
            )
    return support, neighbors


def audit_split(rows: list[dict[str, Any]], split: dict[str, list[int]]) -> dict[str, Any]:
    all_ids = [task for name in SPLIT_ORDER for task in split[name]]
    if sorted(all_ids) != list(range(90)) or len(all_ids) != len(set(all_ids)):
        raise ProtocolError("split task IDs are not a disjoint cover of 0..89")
    if [len(split[name]) for name in SPLIT_ORDER] != [70, 10, 10]:
        raise ProtocolError("split does not have 70/10/10 tasks")
    if any(len({rows[i]["scene"] for i in split[name]}) != 10 for name in HELD_SPLITS):
        raise ProtocolError("a held split does not contain ten distinct scenes")
    split_by_id = {task: name for name in SPLIT_ORDER for task in split[name]}
    owners = _composition_owners(rows, split_by_id)
    support, neighbors = _held_support_and_neighbors(rows, split)
    return {
        "summaries": {name: _summary(rows, split[name]) for name in SPLIT_ORDER},
        "held_role_support": support,
        "held_train_neighbors": neighbors,
        "composition_group_count": len(owners),
        "composition_group_owners": dict(sorted(owners.items())),
        "minimum_train_support_per_held_exact_role_atom": min(
            count for record in support for count in record["train_counts"].values()
        ),
    }


def build_split_artifact(
    rows: list[dict[str, Any]], split: dict[str, list[int]], solver: dict[str, Any], generator_sha256: str, factor_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": solver,
        "generator_sha256": generator_sha256,
        "split_implementation_sha256": sha256_file(Path(__file__)),
        "factor_table_sha256": factor_sha256,
        "constraints": {
            "counts": {"train": 70, "validation": 10, "test": 10},
            "scene_family_quotas_per_held_split": SCENE_FAMILY_QUOTAS,
            "difficulty_quotas_per_held_split": {str(k): v for k, v in DIFFICULTY_QUOTAS.items()},
            "task_family_quotas_per_held_split": TASK_FAMILY_QUOTAS,
            "distinct_scenes_per_held_split": 10,
            "minimum_train_support_per_held_exact_role_atom": 2,
            "composition_policy": (
                "composition groups never cross splits; repeated non-actuation compositions "
                "remain train; repeated actuation groups are indivisible"
            ),
        },
        "objective_order": [
            "minimum weighted validation-test scene/order/role marginal imbalance",
            "minimum weighted deviation from full-pool scene/order/role marginals",
            "seeded specification-hash deterministic tie break",
        ],
        "task_ids": split,
        "audit": audit_split(rows, split),
    }
