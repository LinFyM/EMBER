from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.libero_split import (  # noqa: E402
    SplitResealError,
    audit_split,
    search_split,
)
from ember.contracts import load_contract  # noqa: E402
from ember.libero_task_factors import factor_task  # noqa: E402


def _task(index: int, scene: str, atoms: list[str], composition: str, operations: int = 1) -> dict:
    return {
        "task_index": index,
        "scene": scene,
        "primitive_role_atoms": atoms,
        "composition_signature": composition,
        "difficulty": {"operation_count": operations, "composition_depth": operations},
    }


class LiberoSplitSearchTest(unittest.TestCase):
    def test_audit_requires_two_source_tasks_per_evaluation_role(self) -> None:
        tasks = [
            _task(0, "S1", ["verb:place", "moved_object:bowl"], "a"),
            _task(1, "S1", ["verb:place", "moved_object:bowl"], "b"),
            _task(2, "S1", ["verb:place", "moved_object:bowl"], "c"),
            _task(3, "S2", ["verb:stack"], "d"),
            _task(4, "S2", ["verb:stack"], "e"),
        ]

        passing = audit_split(tasks, source=[0, 1, 3, 4], validation=[2], held_out=[], minimum=2)
        failing = audit_split(tasks, source=[0, 3, 4], validation=[1, 2], held_out=[], minimum=2)

        self.assertTrue(passing["mechanics_valid"])
        self.assertEqual(passing["coverage_violations"], [])
        self.assertFalse(failing["mechanics_valid"])
        self.assertEqual(failing["coverage_violations"][0]["atom"], "moved_object:bowl")

    def test_search_is_seeded_reproducible_and_respects_constraints(self) -> None:
        tasks = []
        for index in range(12):
            tasks.append(
                _task(
                    index,
                    f"S{index // 3}",
                    ["verb:place", f"moved_object:o{index % 3}"],
                    f"composition-{index}",
                    operations=1 + (index % 2),
                )
            )
        prior = {
            "source": list(range(8)),
            "validation": [8, 9],
            "held_out": [10, 11],
        }

        first = search_split(
            tasks,
            prior_split=prior,
            sizes={"source": 8, "validation": 2, "held_out": 2},
            minimum=2,
            seed=17,
            candidate_count=256,
        )
        second = search_split(
            tasks,
            prior_split=prior,
            sizes={"source": 8, "validation": 2, "held_out": 2},
            minimum=2,
            seed=17,
            candidate_count=256,
        )

        self.assertEqual(first, second)
        self.assertTrue(first["audit"]["mechanics_valid"])
        self.assertEqual(len(first["split"]["source"]), 8)
        self.assertEqual(len(first["split"]["validation"]), 2)
        self.assertEqual(len(first["split"]["held_out"]), 2)

    def test_infeasible_search_fails_with_diagnostic(self) -> None:
        tasks = [
            _task(0, "S", ["moved_object:unique"], "a"),
            _task(1, "S", ["moved_object:common"], "b"),
            _task(2, "S", ["moved_object:common"], "c"),
        ]
        prior = {"source": [0], "validation": [1], "held_out": [2]}

        with self.assertRaisesRegex(SplitResealError, "infeasible"):
            search_split(
                tasks,
                prior_split=prior,
                sizes={"source": 1, "validation": 1, "held_out": 1},
                minimum=2,
                seed=7,
                candidate_count=32,
            )


class CheckedInLibero90ResealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(ROOT / "configs" / "phase0.toml")
        cls.record_path = ROOT / "configs" / "libero90_split_reseal.json"
        cls.record_bytes = cls.record_path.read_bytes()
        cls.record = json.loads(cls.record_bytes)

    def test_record_hash_active_split_and_thresholds_are_frozen(self) -> None:
        reseal = self.contract["split_reseal"]
        self.assertEqual(hashlib.sha256(self.record_bytes).hexdigest(), reseal["record_sha256"])
        self.assertEqual(
            self.record["active_split"],
            {
                name: self.contract["splits"][name]
                for name in ("source", "validation", "held_out")
            },
        )
        self.assertEqual(
            self.record["gate_minus_one_thresholds"],
            self.contract["gate_minus_one"]["thresholds"],
        )

    def test_all_ninety_factors_reparse_exactly_and_have_no_privileged_fields(self) -> None:
        self.assertEqual(len(self.record["tasks"]), 90)
        allowed_top_level = {
            "task_index",
            "scene",
            "language",
            "factor_schema",
            "steps",
            "primitive_role_atoms",
            "order_signature",
            "composition_signature",
            "difficulty",
        }
        for task in self.record["tasks"]:
            self.assertEqual(set(task), allowed_top_level)
            self.assertEqual(
                task,
                factor_task(
                    task_index=task["task_index"],
                    scene=task["scene"],
                    language=task["language"],
                ),
            )
        self.assertEqual(self.record["authority"]["numeric_or_privileged_fields_read"], [])

    def test_new_split_passes_and_prior_split_failure_remains_reproducible(self) -> None:
        active = audit_split(self.record["tasks"], **self.record["active_split"], minimum=2)
        prior = audit_split(self.record["tasks"], **self.record["prior_split"], minimum=2)

        self.assertTrue(active["mechanics_valid"])
        self.assertEqual(active["minimum_observed_source_count_for_evaluation_roles"], 2)
        self.assertEqual(active["novel_full_composition_count"], 30)
        self.assertEqual(active["same_scene_source_count"], 30)
        self.assertEqual(active["same_scene_hard_negative_count"], 28)
        self.assertFalse(prior["mechanics_valid"])
        prior_atoms = {failure["atom"] for failure in prior["coverage_violations"]}
        self.assertTrue(
            {
                "verb:stack",
                "moved_object:tomato_sauce",
                "target_relation:under",
                "target_relation:front_of",
                "target_receptacle:wine_rack",
            }.issubset(prior_atoms)
        )


if __name__ == "__main__":
    unittest.main()
