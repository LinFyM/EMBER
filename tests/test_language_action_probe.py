from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.language_action_probe import (  # noqa: E402
    LanguageActionProbeError,
    compare_action_plans,
    decide_action_probe,
    load_action_spec,
    validate_action_spec,
)
from ember.specification_probe import load_specification_spec  # noqa: E402


class LanguageActionProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pilot = load_specification_spec(
            ROOT / "configs" / "gate_minus1_specification_pilot.toml"
        )
        cls.spec = load_action_spec(
            ROOT / "configs" / "gate_minus1_language_action_probe.toml",
            cls.pilot,
        )

    def test_checked_in_probe_reuses_fixed_batch_and_is_diagnostic_only(self) -> None:
        self.assertEqual(self.spec["planned_action_steps"], 10)
        self.assertEqual(self.spec["primary_comparison"], "correct_vs_swapped")
        self.assertEqual(self.spec["minimum_substantive_fraction"], 0.8)
        self.assertEqual(self.spec["substantive_plan_max_abs_delta"], 0.01)
        self.assertFalse(self.spec["interpretation"]["gate_decision_authorized"])
        self.assertEqual(self.pilot["batch_size"], 8)
        self.assertTrue(self.pilot["use_async_envs"])
        pilot_path = ROOT / "configs" / "gate_minus1_specification_pilot.toml"
        self.assertEqual(
            hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
            self.spec["pilot_config_sha256"],
        )
        self.assertGreater(
            self.spec["substantive_plan_max_abs_delta"],
            4 * self.spec["known_cross_batch_max_abs_delta"],
        )

    def test_probe_fails_closed_on_batch_or_prompt_contract_drift(self) -> None:
        changed_pilot = copy.deepcopy(self.pilot)
        changed_pilot["batch_size"] = 16
        with self.assertRaisesRegex(LanguageActionProbeError, "batch"):
            validate_action_spec(self.spec, changed_pilot)

        changed = copy.deepcopy(self.spec)
        changed["conditions"] = ["correct", "swapped"]
        with self.assertRaisesRegex(LanguageActionProbeError, "conditions"):
            validate_action_spec(changed, self.pilot)

    def test_action_comparison_counts_substantive_episode_plans(self) -> None:
        correct = np.zeros((8, 10, 7), dtype=np.float32)
        swapped = correct.copy()
        swapped[:7, :, 0] = 0.02

        report = compare_action_plans(
            correct,
            swapped,
            atol=self.spec["action_atol"],
            rtol=self.spec["action_rtol"],
            substantive_delta=self.spec["substantive_plan_max_abs_delta"],
        )

        self.assertEqual(report["shape"], [8, 10, 7])
        self.assertEqual(report["differing_episodes"], 7)
        self.assertEqual(report["substantive_episodes"], 7)
        self.assertEqual(report["substantive_fraction"], 0.875)

    def test_decision_requires_repeat_stability_before_language_path(self) -> None:
        primary = {"substantive_fraction": 1.0}

        passed = decide_action_probe(self.spec, repeat_stable=True, primary=primary)
        stopped = decide_action_probe(self.spec, repeat_stable=False, primary=primary)

        self.assertEqual(passed["status"], "language_action_path_present")
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(stopped["reason"], "correct_repeat_instability")
        self.assertFalse(passed["gate_decision_authorized"])


if __name__ == "__main__":
    unittest.main()
