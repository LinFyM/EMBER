from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.specification_probe import (  # noqa: E402
    SpecificationProbeError,
    ResetAuditEnv,
    apply_prompt_override,
    decide_pilot,
    load_specification_spec,
    paired_gap_summary,
    resolve_prompt,
    validate_specification_spec,
    _validate_contract_alignment,
)
from ember.contracts import load_contract  # noqa: E402


class FakeVectorEnv:
    def __init__(self) -> None:
        self.values = {
            "task": ["TASK_A", "TASK_A"],
            "task_description": ["correct instruction", "correct instruction"],
            "init_state_id": [0, 1],
        }

    def call(self, name: str):
        return tuple(self.values[name])

    def set_attr(self, name: str, values: list[str]) -> None:
        self.values[name] = list(values)

    def reset(self, *, seed: list[int]):
        self.values["init_state_id"] = [2, 3]
        return {"pixels": []}, {}


class SpecificationProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_minus1_specification_pilot.toml"
        cls.spec = load_specification_spec(cls.path)

    def test_checked_in_pilot_is_fixed_batch_and_diagnostic_only(self) -> None:
        self.assertEqual(self.spec["task_ids"], [0, 1])
        self.assertEqual(self.spec["batch_size"], self.spec["episodes_per_task"])
        self.assertTrue(self.spec["use_async_envs"])
        self.assertFalse(self.spec["pilot_advancement"]["gate_decision_authorized"])
        self.assertEqual(
            self.spec["conditions"], ["correct", "no_spec", "scene_only", "swapped"]
        )

    def test_checkpoint_role_and_thresholds_match_phase0_contract(self) -> None:
        contract = load_contract(ROOT / "configs" / "phase0.toml")
        _validate_contract_alignment(self.spec, contract)

    def test_pair_map_must_be_involutive_and_cover_tasks(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["hard_negative_pairs"][0]["right"] = 2
        with self.assertRaisesRegex(SpecificationProbeError, "pair"):
            validate_specification_spec(changed)

    def test_batch_shape_must_not_change_between_arms(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["episodes_per_task"] = 16
        with self.assertRaisesRegex(SpecificationProbeError, "one fixed batch"):
            validate_specification_spec(changed)

    def test_prompt_conditions_are_exact_and_swapped_within_pair(self) -> None:
        languages = {0: "instruction zero", 1: "instruction one"}
        self.assertEqual(resolve_prompt(self.spec, 0, "correct", languages), "instruction zero")
        self.assertEqual(resolve_prompt(self.spec, 0, "no_spec", languages), "")
        self.assertEqual(
            resolve_prompt(self.spec, 0, "scene_only", languages),
            self.spec["scene_only_prompt"],
        )
        self.assertEqual(resolve_prompt(self.spec, 0, "swapped", languages), "instruction one")

    def test_prompt_override_preserves_environment_task_identity(self) -> None:
        env = FakeVectorEnv()
        report = apply_prompt_override(env, "neutral", batch_size=2)

        self.assertEqual(report["task_before"], ["TASK_A", "TASK_A"])
        self.assertEqual(report["task_after"], report["task_before"])
        self.assertEqual(report["prompt_after"], ["neutral", "neutral"])
        self.assertTrue(report["mechanically_valid"])

    def test_reset_audit_records_the_upstream_seed_and_first_mapping(self) -> None:
        audited = ResetAuditEnv(FakeVectorEnv())
        audited.reset(seed=[5100, 5101])

        self.assertEqual(
            audited.reset_events,
            [{"before": [0, 1], "after": [2, 3], "seeds": [5100, 5101]}],
        )

    def test_paired_gap_bootstrap_is_deterministic(self) -> None:
        arms = [
            {"task_id": 0, "condition": "correct", "successes": [True, True, False, True]},
            {"task_id": 0, "condition": "swapped", "successes": [False, False, False, True]},
            {"task_id": 1, "condition": "correct", "successes": [True, False, True, False]},
            {"task_id": 1, "condition": "swapped", "successes": [False, False, True, False]},
        ]
        first = paired_gap_summary(
            arms, left="correct", right="swapped", seed=7, replicates=500
        )
        second = paired_gap_summary(
            arms, left="correct", right="swapped", seed=7, replicates=500
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["gap_pp"], 37.5)
        self.assertEqual(first["task_count"], 2)

    def test_pilot_stops_if_any_correct_arm_has_zero_success(self) -> None:
        arms = [
            {"task_id": 0, "condition": "correct", "successes": [True, False]},
            {"task_id": 1, "condition": "correct", "successes": [False, False]},
        ]
        decision = decide_pilot(self.spec, arms, mechanics_valid=True)

        self.assertEqual(decision["status"], "stopped")
        self.assertEqual(decision["reason"], "correct_arm_zero_success")
        self.assertFalse(decision["gate_decision_authorized"])


if __name__ == "__main__":
    unittest.main()
