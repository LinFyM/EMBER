from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.counterfactual_goal_probe import (  # noqa: E402
    CounterfactualGoalProbeError,
    assess_goal_switch,
    load_probe_spec,
    validate_probe_spec,
)


class CounterfactualGoalProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_probe_spec(
            ROOT / "configs" / "gate_minus1_same_init_goal_probe.toml"
        )

    def test_checked_in_probe_is_source_only_and_diagnostic(self) -> None:
        self.assertEqual(self.spec["task_ids"], [3, 4])
        self.assertEqual(self.spec["source_task_ids"], [3, 4])
        self.assertEqual(self.spec["demonstration_indices"], list(range(8)))
        self.assertEqual(self.spec["initial_state_indices"], list(range(8)))
        self.assertEqual(self.spec["counterfactual_switch_threshold"], 0.8)
        self.assertFalse(self.spec["claim_boundary"]["gate_decision_authorized"])

    def test_checked_in_pair_matches_the_permanent_split_seal(self) -> None:
        seal_path = ROOT / "configs" / "libero90_split_reseal.json"
        payload = seal_path.read_bytes()
        seal = json.loads(payload)

        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.spec["split_seal_sha256"])
        self.assertTrue(set(self.spec["task_ids"]).issubset(seal["active_split"]["source"]))
        self.assertEqual(len({task["goal_sha256"] for task in self.spec["tasks"]}), 2)

    def test_surface_and_native_evaluator_are_fail_closed(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["success_evaluator"] = "custom_goal_heuristic"
        with self.assertRaisesRegex(CounterfactualGoalProbeError, "native"):
            validate_probe_spec(changed)

        changed = copy.deepcopy(self.spec)
        changed["surface"] = "libero90_held"
        with self.assertRaisesRegex(CounterfactualGoalProbeError, "source"):
            validate_probe_spec(changed)

    def test_pair_must_be_distinct_and_fully_declared_source(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["task_ids"] = [3, 3]
        with self.assertRaisesRegex(CounterfactualGoalProbeError, "distinct"):
            validate_probe_spec(changed)

        changed = copy.deepcopy(self.spec)
        changed["source_task_ids"] = [3]
        with self.assertRaisesRegex(CounterfactualGoalProbeError, "source"):
            validate_probe_spec(changed)

    def test_exact_same_state_and_bidirectional_specificity_pass(self) -> None:
        initial_rows = [
            {
                "state_identity_exact": True,
                "success_by_task": {"3": False, "4": False},
            }
            for _ in range(8)
        ]
        terminal_rows = []
        for origin in (3, 4):
            for _ in range(8):
                terminal_rows.append(
                    {
                        "origin_task_id": origin,
                        "state_identity_exact": True,
                        "success_by_task": {
                            "3": origin == 3,
                            "4": origin == 4,
                        },
                    }
                )

        report = assess_goal_switch(self.spec, initial_rows, terminal_rows)

        self.assertEqual(report["status"], "mechanics_pass")
        self.assertEqual(report["minimum_switch_fraction"], 1.0)
        self.assertTrue(report["initial_state_contract_passed"])
        self.assertFalse(report["gate_decision_authorized"])

    def test_identity_mismatch_fails_even_if_goal_matrix_is_specific(self) -> None:
        initial_rows = [
            {
                "state_identity_exact": index != 0,
                "success_by_task": {"3": False, "4": False},
            }
            for index in range(8)
        ]
        terminal_rows = [
            {
                "origin_task_id": origin,
                "state_identity_exact": True,
                "success_by_task": {"3": origin == 3, "4": origin == 4},
            }
            for origin in (3, 4)
            for _ in range(8)
        ]

        report = assess_goal_switch(self.spec, initial_rows, terminal_rows)

        self.assertEqual(report["status"], "mechanics_failed")
        self.assertFalse(report["initial_state_contract_passed"])


if __name__ == "__main__":
    unittest.main()
