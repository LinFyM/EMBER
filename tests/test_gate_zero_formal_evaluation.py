from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from ember.gate_zero_task_local_rl.formal_evaluation import (
    aggregate_formal_rows,
    compatible_recovery_authorities,
    load_formal_evaluation_spec,
    rollout_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class FormalEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, cls.evidence = load_formal_evaluation_spec(
            ROOT / "configs/gate_zero_formal_development_evaluation.toml",
            repo_root=ROOT,
        )

    def test_contract_freezes_n32_multi_seed_h16_h50_evaluation(self) -> None:
        evaluation = self.spec["evaluation"]
        self.assertEqual(evaluation["rollouts_per_task_arm"], 32)
        self.assertEqual(len(evaluation["policy_rng_seeds"]), 4)
        self.assertEqual(evaluation["required_training_seeds"], [2026071830, 2026072030])
        self.assertEqual(evaluation["execution_horizons"], [16, 50])
        self.assertTrue(evaluation["fixed_initializations_are_evaluated_once"])
        self.assertFalse(self.spec["authority"]["held_numeric_access"])

    def test_rollout_rows_preserve_pairing_and_true_time_to_success(self) -> None:
        rollout = {
            "official_rollout_init_state_indices": list(range(40, 48)),
            "seeds": list(range(5800, 5808)),
            "successes": [True, False] * 4,
            "max_rewards": [1.0, 0.0] * 4,
            "time_to_success": [17, None] * 4,
        }
        rows = rollout_rows(
            rollout=rollout,
            task_id=3,
            arm="zero_init_rl",
            training_seed=2026071830,
            policy_rng_seed=2026071836,
            execution_horizon=16,
            init_state_hashes={index: f"{index:064x}" for index in range(40, 48)},
            action_drift_to_base=0.01,
            action_drift_to_initialization=0.01,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["time_to_success"], 17)
        self.assertIsNone(rows[1]["time_to_success"])
        self.assertEqual(rows[0]["physical_init_state_index"], 40)

    def test_reference_seed_legacy_authority_is_the_only_allowed_omission(self) -> None:
        expected = {"task_id": 3, "training_seed": 2026071830}
        self.assertEqual(
            compatible_recovery_authorities(
                expected,
                {"task_id": 3},
                spec=self.spec,
                training_seed=2026071830,
            ),
            {"task_id": 3},
        )
        with self.assertRaises(Exception):
            compatible_recovery_authorities(
                expected,
                {"task_id": 4},
                spec=self.spec,
                training_seed=2026071830,
            )

    def test_aggregate_uses_fixed_baselines_once_and_two_training_seeds(self) -> None:
        rows = []
        arms = ("frozen_base", "supervised_lora", "zero_init_rl", "supervised_init_rl")
        policy_seeds = self.spec["evaluation"]["policy_rng_seeds"]
        for task in (3, 4):
            for horizon in (16, 50):
                for arm in arms:
                    seeds = [None] if arm in arms[:2] else [2026071830, 2026072030]
                    for training_seed in seeds:
                        for policy_offset, policy_seed in enumerate(policy_seeds):
                            for offset, state in enumerate(range(40, 48)):
                                baseline = offset < 2
                                success = baseline
                                if arm in {"zero_init_rl", "supervised_init_rl"}:
                                    success = offset < 6
                                rows.append(
                                    {
                                        "surface": "development",
                                        "task_id": task,
                                        "arm": arm,
                                        "training_seed": training_seed,
                                        "policy_rng_seed": policy_seed,
                                        "evaluator_seed": 5800 + offset,
                                        "physical_init_state_index": state,
                                        "physical_init_state_sha256": f"{state:064x}",
                                        "execution_horizon": horizon,
                                        "success": success,
                                        "grasp": None,
                                        "correct_object_or_region": None,
                                        "drawer_closed": None,
                                        "time_to_success": 10 if success else None,
                                        "progress_fraction": float(success),
                                        "action_drift_to_base": 0.01,
                                        "action_drift_to_initialization": 0.005,
                                    }
                                )
        result = aggregate_formal_rows(rows, spec=self.spec, evidence=self.evidence)
        self.assertEqual(result["validation"]["minimum_count"], 32)
        self.assertEqual(result["validation"]["training_seed_count"], 2)
        self.assertTrue(result["development_candidate_supported"])


if __name__ == "__main__":
    unittest.main()
