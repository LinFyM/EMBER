from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_contract import GateZeroContractError, load_gate_zero_contract  # noqa: E402


class GateZeroContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "configs" / "gate_zero_oracle_pilot.toml"

    def test_checked_in_contract_is_source_only_and_frozen_before_outcomes(self) -> None:
        spec = load_gate_zero_contract(self.path, ROOT / "configs" / "phase0.toml")

        self.assertEqual(spec["status"], "predeclared_before_source_policy_outcomes")
        self.assertEqual(spec["data"]["task_ids"], [3, 4])
        self.assertEqual(spec["access"]["oracle_support"], [28, 39])
        self.assertEqual(spec["access"]["functional_query"], [40, 45])
        self.assertEqual(spec["access"]["locked_source_report"], [46, 49])
        self.assertEqual(spec["access"]["fully_pristine_all_fields"], [48, 49])
        self.assertEqual(spec["access"]["forbidden_numeric_splits"], ["validation", "held_out"])
        self.assertFalse(spec["recovery"]["writer_authorized_by_pilot"])

    def test_primary_oracle_is_exactly_four_rank8_action_expert_matrices(self) -> None:
        spec = load_gate_zero_contract(self.path, ROOT / "configs" / "phase0.toml")

        self.assertEqual(spec["oracle"]["rank"], 8)
        self.assertEqual(spec["oracle"]["alpha"], 8)
        self.assertIs(spec["oracle"]["init_lora_weights"], True)
        self.assertEqual(spec["oracle"]["expected_trainable_parameters"], 40320)
        self.assertEqual(len(spec["oracle"]["target_modules"]), 4)
        self.assertTrue(all("lm_expert.layers." in name for name in spec["oracle"]["target_modules"]))
        self.assertNotIn("action_out_proj", " ".join(spec["oracle"]["target_modules"]))

    def test_report_is_sealed_from_checkpoint_selection(self) -> None:
        spec = load_gate_zero_contract(self.path, ROOT / "configs" / "phase0.toml")

        self.assertFalse(spec["oracle"]["selection"]["report_access_before_selection_freeze"])
        self.assertTrue(spec["report"]["no_selection_after_report_access"])
        self.assertEqual(spec["oracle"]["selection"]["episode_bounds"], [40, 45])
        self.assertEqual(spec["report"]["offline_episode_bounds"], [46, 49])

    def test_loader_rejects_episode_overlap(self) -> None:
        text = self.path.read_text(encoding="utf-8").replace(
            "functional_query = [40, 45]", "functional_query = [39, 45]"
        )
        with self.assertRaisesRegex(GateZeroContractError, "episode partitions"):
            load_gate_zero_contract(text, ROOT / "configs" / "phase0.toml", from_text=True)


if __name__ == "__main__":
    unittest.main()
