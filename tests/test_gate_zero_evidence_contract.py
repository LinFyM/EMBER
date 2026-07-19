from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ember.gate_zero_evidence import (
    GateZeroEvidenceError,
    canonical_confirmation_candidates,
    deterministic_state_partition,
    load_gate_zero_evidence_spec,
    paired_binary_summary,
    validate_bound_authority,
    validate_evaluation_records,
)


ROOT = Path(__file__).resolve().parents[1]


class GateZeroEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_evidence_repair.toml"
        cls.split_path = ROOT / "configs" / "libero90_split_reseal.json"
        cls.spec = load_gate_zero_evidence_spec(cls.path, cls.split_path)
        cls.split = json.loads(cls.split_path.read_text(encoding="utf-8"))

    def test_gate_minus_one_is_passed_with_preserved_residuals(self) -> None:
        resolution = self.spec["gate_minus_one_resolution"]
        self.assertEqual(resolution["status"], "passed_with_residuals")
        self.assertAlmostEqual(resolution["ordered_balanced_accuracy"], 19 / 24)
        self.assertEqual(resolution["bidirectional_pairs_correct"], 15)
        self.assertEqual(resolution["bidirectional_pairs_total"], 24)
        self.assertEqual(resolution["original_content_threshold"], 0.80)
        self.assertTrue(resolution["drop_last_residual_preserved"])
        self.assertFalse(resolution["additional_compute_to_reach_threshold_authorized"])

    def test_candidate_pool_is_derived_from_source_spec_only(self) -> None:
        candidates = canonical_confirmation_candidates(
            self.split, development_task_ids=[3, 4]
        )
        self.assertEqual(candidates, [6, 9, 16, 20, 23, 33, 39, 46, 63])
        self.assertEqual(
            self.spec["confirmation_selection"]["candidate_task_ids"], candidates
        )
        source = set(self.split["active_split"]["source"])
        self.assertTrue(set(candidates) <= source)
        self.assertFalse(set(candidates) & {3, 4})

    def test_state_partition_is_seeded_disjoint_and_hash_bound(self) -> None:
        first = deterministic_state_partition(task_id=6, seed=20260719)
        second = deterministic_state_partition(task_id=6, seed=20260719)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 32)
        self.assertEqual(len(first["development"]), 16)
        self.assertEqual(len(first["reserve"]), 2)
        self.assertEqual(
            set(first["train"]) | set(first["development"]) | set(first["reserve"]),
            set(range(50)),
        )
        self.assertFalse(set(first["train"]) & set(first["development"]))
        self.assertEqual(len(first["sha256"]), 64)

    def test_n8_is_smoke_only_and_scientific_minima_are_enforced(self) -> None:
        evaluation = self.spec["evaluation"]
        self.assertEqual(evaluation["smoke_rollouts_per_task_arm"], 8)
        self.assertFalse(evaluation["smoke_may_select_or_reject_mechanism"])
        self.assertGreaterEqual(evaluation["minimum_rollouts_per_task_arm"], 32)
        self.assertGreaterEqual(evaluation["minimum_policy_rng_seeds"], 2)
        self.assertGreaterEqual(evaluation["minimum_training_seeds"], 2)
        self.assertEqual(evaluation["preferred_training_seeds"], 3)
        self.assertEqual(evaluation["primary_execution_horizon"], 16)
        self.assertEqual(evaluation["deployment_robustness_horizon"], 50)

    def test_algorithm_names_actual_pilot_and_required_fpo_core(self) -> None:
        pilot = self.spec["custom_pilot"]
        self.assertEqual(pilot["name"], "custom_chunk_level_flow_loss_ppo_pilot")
        self.assertEqual(pilot["flow_samples_averaged_before_ratio"], 8)
        self.assertNotIn("flow_sample_group_size", pilot)
        faithful = self.spec["faithful_fpo_plus_core"]
        self.assertTrue(faithful["required_before_ordinary_rl_negative_claim"])
        self.assertEqual(faithful["ratio_granularity"], "per_flow_sample")
        self.assertEqual(faithful["cfm_loss_average_group_size"], 1)
        self.assertTrue(faithful["modified_huber_matches_mse_below_delta"])
        self.assertEqual(faithful["old_cfm_loss_clamp"], 4.0)
        self.assertEqual(faithful["log_ratio_clamp"], 5.0)

    def _records(self, *, count: int = 32, training_seeds=(71, 72)) -> list[dict]:
        rows = []
        for training_seed in training_seeds:
            for task_id in (3, 4):
                for horizon in (16, 50):
                    for arm in self.spec["evaluation"]["arms"]:
                        for episode in range(count):
                            rows.append(
                                {
                                    "surface": "development",
                                    "task_id": task_id,
                                    "arm": arm,
                                    "training_seed": training_seed,
                                    "policy_rng_seed": 100 + episode % 4,
                                    "evaluator_seed": 1000 + episode,
                                    "physical_init_state_index": episode % 16,
                                    "physical_init_state_sha256": f"{episode % 16:064x}",
                                    "execution_horizon": horizon,
                                    "success": bool((episode + task_id) % 3),
                                    "grasp": bool((episode + task_id) % 2),
                                    "correct_object_or_region": bool(
                                        (episode + task_id) % 3
                                    ),
                                    "drawer_closed": None,
                                    "time_to_success": 50 if (episode + task_id) % 3 else None,
                                    "progress_fraction": 1.0 if (episode + task_id) % 3 else 0.5,
                                    "action_drift_to_base": 0.01,
                                    "action_drift_to_initialization": 0.005,
                                }
                            )
        return rows

    def test_evaluation_records_require_rollouts_policy_and_training_seeds(self) -> None:
        summary = validate_evaluation_records(self._records(), self.spec)
        self.assertEqual(summary["minimum_count"], 32)
        self.assertEqual(summary["training_seed_count"], 2)
        self.assertEqual(summary["policy_rng_seed_count_minimum"], 4)
        self.assertEqual(summary["primary_execution_horizon"], 16)
        self.assertEqual(summary["deployment_robustness_horizon"], 50)

        with self.assertRaises(GateZeroEvidenceError):
            validate_evaluation_records(self._records(count=8), self.spec)
        with self.assertRaises(GateZeroEvidenceError):
            validate_evaluation_records(self._records(training_seeds=(71,)), self.spec)
        changed = self._records()
        for row in changed:
            row["policy_rng_seed"] = 100
        with self.assertRaises(GateZeroEvidenceError):
            validate_evaluation_records(changed, self.spec)
        unpaired = self._records()
        unpaired[0]["evaluator_seed"] += 10_000
        with self.assertRaises(GateZeroEvidenceError):
            validate_evaluation_records(unpaired, self.spec)

    def test_paired_summary_has_bootstrap_and_exact_discordant_interval(self) -> None:
        left = [True] * 12 + [False] * 20
        right = [True] * 8 + [False] * 24
        result = paired_binary_summary(
            left, right, bootstrap_seed=7, bootstrap_replicates=2000
        )
        self.assertEqual(result["episodes"], 32)
        self.assertEqual(result["paired_wins"], 4)
        self.assertEqual(result["paired_losses"], 0)
        self.assertEqual(result["paired_ties"], 28)
        self.assertAlmostEqual(result["net_gain_pp"], 12.5)
        self.assertEqual(result["bootstrap_replicates"], 2000)
        self.assertEqual(result["exact_discordant_trials"], 4)
        lower, upper = result["exact_conditional_win_rate_ci95"]
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLess(lower, upper)

    def test_surface_overlap_and_contract_mutation_fail_closed(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["confirmation_selection"]["candidate_task_ids"] = [3, 6]
        from ember.gate_zero_evidence import validate_gate_zero_evidence_spec

        with self.assertRaises(GateZeroEvidenceError):
            validate_gate_zero_evidence_spec(changed, self.split)

    def test_bound_repository_and_external_evidence_are_hash_checked(self) -> None:
        changed = copy.deepcopy(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root = root / "repo"
            output_root = root / "output"
            bindings = (
                (repo_root, "horizon_coverage_contract", b"contract"),
                (output_root, "horizon_coverage_stage24_result", b"stage24"),
                (output_root, "video_information_result", b"video"),
            )
            for base, prefix, content in bindings:
                path = base / f"{prefix}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                changed["authority"][f"{prefix}_relative_path"] = path.name
                changed["authority"][f"{prefix}_sha256"] = hashlib.sha256(
                    content
                ).hexdigest()
            verified = validate_bound_authority(
                changed, repo_root=repo_root, output_root=output_root
            )
            self.assertEqual(len(verified), 3)
            (output_root / "video_information_result.json").write_bytes(b"changed")
            with self.assertRaises(GateZeroEvidenceError):
                validate_bound_authority(
                    changed, repo_root=repo_root, output_root=output_root
                )


if __name__ == "__main__":
    unittest.main()
