from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_support.candidate_step_diagnostic import (  # noqa: E402
    GateZeroCandidateStepDiagnosticError,
    assigned_candidate_step_arms,
    decide_candidate_step_diagnostic,
    load_candidate_step_diagnostic_spec,
    validate_candidate_step_diagnostic_spec,
)


class GateZeroMatureLoraCandidateStepDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_mature_lora_candidate_step_diagnostic.toml"
        cls.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        cls.phase0 = ROOT / "configs" / "phase0.toml"
        cls.fit = ROOT / "configs" / "gate_zero_mature_lora_lr_recovery.toml"
        cls.proposal_a = ROOT / "configs" / "gate_zero_mature_lora_headroom_screen.toml"
        cls.spec = load_candidate_step_diagnostic_spec(
            cls.path,
            gate_zero_path=cls.gate_zero,
            phase0_path=cls.phase0,
            fit_path=cls.fit,
            proposal_a_path=cls.proposal_a,
        )

    def _arm(self, task_id: int, step: int, successes: list[bool]) -> dict:
        return {
            "task_id": task_id,
            "condition": f"step{step}",
            "candidate_step": step,
            "mechanics_valid": True,
            "official_rollout_init_state_indices": list(range(40, 48)),
            "seeds": list(range(5800, 5808)),
            "successes": successes,
        }

    def test_contract_reuses_same_tasks_and_never_continues_training(self) -> None:
        self.assertEqual(self.spec["task_ids"], [3, 4])
        self.assertEqual(self.spec["candidate_steps"], [500, 750])
        self.assertFalse(self.spec["authority"]["continuation_past_step1000"])
        self.assertEqual(self.spec["decision"]["minimum_median_success_gain_pp"], 15.0)
        self.assertEqual(self.spec["decision"]["minimum_positive_task_count"], 2)
        self.assertFalse(self.spec["proposal_a_failure"]["gate_zero_authorized"])

    def test_two_gpu_assignment_has_no_duplicate_or_idle_arm(self) -> None:
        left = assigned_candidate_step_arms(rank=0, world_size=2, spec=self.spec)
        right = assigned_candidate_step_arms(rank=1, world_size=2, spec=self.spec)
        self.assertEqual(left, [(3, 500), (3, 750)])
        self.assertEqual(right, [(4, 500), (4, 750)])
        self.assertEqual(set(left) | set(right), {(3, 500), (3, 750), (4, 500), (4, 750)})

    def test_original_positive_rule_selects_best_then_earliest(self) -> None:
        # Frozen A bases are 3/8 for both tasks. Step 500 gains +2/+1 and
        # step 750 gains +1/+2; both have aggregate +3 and pass the original
        # median-15pp/two-positive-tasks rule, so the earlier step wins.
        arms = [
            self._arm(3, 500, [True] * 5 + [False] * 3),
            self._arm(3, 750, [True] * 4 + [False] * 4),
            self._arm(4, 500, [True] * 4 + [False] * 4),
            self._arm(4, 750, [True] * 5 + [False] * 3),
        ]
        decision = decide_candidate_step_diagnostic(self.spec, arms)
        self.assertEqual(decision["status"], "earlier_candidate_step_selected_for_fresh_recovery_gate")
        self.assertEqual(decision["selected_step"], 500)
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_one_task_nonpositive_cannot_select_checkpoint(self) -> None:
        arms = [
            self._arm(3, 500, [True] * 4 + [False] * 4),
            self._arm(3, 750, [True] * 3 + [False] * 5),
            self._arm(4, 500, [True] * 3 + [False] * 5),
            self._arm(4, 750, [True] * 5 + [False] * 3),
        ]
        decision = decide_candidate_step_diagnostic(self.spec, arms)
        self.assertEqual(decision["status"], "candidate_step_magnitude_recovery_not_supported")
        self.assertIsNone(decision["selected_step"])

    def test_seen_a_numbers_cannot_change_threshold(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["decision"]["minimum_median_success_gain_pp"] = 12.5
        with self.assertRaises(GateZeroCandidateStepDiagnosticError):
            validate_candidate_step_diagnostic_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                proposal_a_path=self.proposal_a,
            )

    def test_launcher_has_two_gpu_bound_and_no_training_resume(self) -> None:
        launcher = (ROOT / "scripts" / "run_gate_zero_candidate_step_diagnostic.sh").read_text()
        self.assertIn("nproc-per-node=2", launcher)
        self.assertNotIn("--resume", launcher)
        self.assertNotIn("stop-after-step", launcher)
        self.assertIn("minimum_free_memory_mib", launcher)


if __name__ == "__main__":
    unittest.main()
