from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_support.capacity_closed_loop import (  # noqa: E402
    GateZeroCapacityClosedLoopError,
    assigned_capacity_arms,
    decide_capacity_closed_loop,
    load_capacity_closed_loop_spec,
    validate_capacity_closed_loop_spec,
)


class GateZeroActionExpertCapacityClosedLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ROOT / "configs/gate_zero_action_expert_capacity_closed_loop.toml"
        cls.gate_zero = ROOT / "configs/gate_zero_oracle_pilot.toml"
        cls.phase0 = ROOT / "configs/phase0.toml"
        cls.competence = ROOT / "configs/gate_zero_source_competence.toml"
        cls.fit = ROOT / "configs/gate_zero_mature_action_expert_lr_recovery.toml"
        cls.spec = load_capacity_closed_loop_spec(
            cls.config,
            gate_zero_path=cls.gate_zero,
            phase0_path=cls.phase0,
            competence_path=cls.competence,
            fit_path=cls.fit,
        )

    def _arms(self, task3: list[bool], task4: list[bool]) -> list[dict]:
        rollout = self.spec["rollout"]
        return [
            {
                "task_id": task_id,
                "condition": self.spec["variant"],
                "successes": successes,
                "seeds": list(range(rollout["seed_start"], rollout["seed_start"] + 8)),
                "official_rollout_init_state_indices": rollout["init_state_indices"],
                "mechanics_valid": True,
            }
            for task_id, successes in ((3, task3), (4, task4))
        ]

    def test_contract_is_nonmatched_source_only_and_never_authorizes(self) -> None:
        self.assertEqual(self.spec["task_ids"], [3, 4])
        self.assertEqual(self.spec["candidate_step"], 1000)
        self.assertEqual(self.spec["candidate_evidence"]["trainable_parameters"], 99_880_992)
        self.assertFalse(self.spec["candidate_evidence"]["matched_lora_baseline"])
        self.assertFalse(self.spec["authority"]["validation_numeric_access"])
        self.assertFalse(self.spec["authority"]["held_numeric_access"])

    def test_contract_hash_or_rollout_mutation_fails_closed(self) -> None:
        for key, value in (
            ("fit_contract_sha256", "0" * 64),
            ("proposal_a_result_sha256", "0" * 64),
            ("signed_ratio_result_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(self.spec)
            changed["authority"][key] = value
            with self.assertRaises(GateZeroCapacityClosedLoopError):
                validate_capacity_closed_loop_spec(
                    changed,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    fit_path=self.fit,
                )
        changed = copy.deepcopy(self.spec)
        changed["rollout"]["seed_start"] += 1
        with self.assertRaises(GateZeroCapacityClosedLoopError):
            validate_capacity_closed_loop_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                competence_path=self.competence,
                fit_path=self.fit,
            )

    def test_two_rank_assignment_has_one_task_per_rank(self) -> None:
        self.assertEqual(assigned_capacity_arms(rank=0, world_size=2, spec=self.spec), [3])
        self.assertEqual(assigned_capacity_arms(rank=1, world_size=2, spec=self.spec), [4])
        self.assertEqual(assigned_capacity_arms(rank=0, world_size=1, spec=self.spec), [3, 4])

    def test_positive_upper_bound_is_diagnostic_not_gate_pass(self) -> None:
        decision = decide_capacity_closed_loop(
            self.spec,
            self._arms(
                [True, True, False, False, True, False, True, False],
                [True, True, False, False, True, False, True, True],
            ),
        )
        self.assertEqual(
            decision["status"], "nonmatched_action_expert_capacity_behavioral_signal_present"
        )
        self.assertEqual(decision["paired_net_wins_by_task"], {"3": 1, "4": 2})
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_missing_task_improvement_fails_without_changing_threshold(self) -> None:
        decision = decide_capacity_closed_loop(
            self.spec,
            self._arms(
                [True, False, False, False, True, False, True, False],
                [True, True, False, False, True, False, True, True],
            ),
        )
        self.assertEqual(
            decision["status"], "nonmatched_action_expert_capacity_behavioral_signal_absent"
        )
        self.assertFalse(decision["capacity_signal_present"])

    def test_launcher_reuses_two_gpu_canonical_evaluator(self) -> None:
        text = (ROOT / "scripts/run_gate_zero_action_expert_capacity_closed_loop.sh").read_text()
        self.assertIn("--nproc-per-node=2", text)
        self.assertIn("ember.gate_zero_support.capacity_closed_loop", text)
        self.assertIn("gpu_telemetry_", text)
        self.assertNotIn("validation", text.lower())
        self.assertNotIn("held", text.lower())


if __name__ == "__main__":
    unittest.main()
