from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_query_action_alignment import (  # noqa: E402
    classify_action_alignment,
    classify_multi_seed_action_alignment,
    load_query_action_alignment_spec,
)


class GateZeroQueryActionAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_query_action_alignment_audit.toml"
        self.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"
        self.lora_fit = ROOT / "configs" / "gate_zero_mature_lora_lr_recovery.toml"
        self.action_fit = ROOT / "configs" / "gate_zero_mature_action_expert_lr_recovery.toml"
        self.capacity = ROOT / "configs" / "gate_zero_action_expert_capacity_closed_loop.toml"
        self.robustness = ROOT / "configs" / "gate_zero_query_action_alignment_robustness.toml"

    def test_checked_in_contract_is_source_only_and_non_authorizing(self) -> None:
        spec = load_query_action_alignment_spec(
            self.config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
            lora_fit_path=self.lora_fit,
            action_fit_path=self.action_fit,
            capacity_path=self.capacity,
        )

        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["query"]["episode_bounds"], [40, 45])
        self.assertEqual(spec["query"]["anchor_count_per_task"], 48)
        self.assertEqual(spec["query"]["new_environment_rollout_episodes"], 0)
        self.assertFalse(spec["decision"]["may_authorize_gate_zero"])
        self.assertFalse(spec["decision"]["may_authorize_writer"])

    def test_robustness_contract_freezes_four_noise_draws_without_rollout(self) -> None:
        spec = load_query_action_alignment_spec(
            self.robustness,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
            lora_fit_path=self.lora_fit,
            action_fit_path=self.action_fit,
            capacity_path=self.capacity,
        )

        self.assertEqual(
            spec["query"]["inference_noise_seeds"],
            [2026071835, 2026071935, 2026072035, 2026072135],
        )
        self.assertEqual(spec["query"]["new_environment_rollout_episodes"], 0)
        self.assertEqual(spec["decision"]["minimum_consistent_worsening_draws"], 3)

    def test_classification_distinguishes_surrogate_temporal_and_mixed_failures(self) -> None:
        base = {3: 1.0, 4: 2.0}
        self.assertEqual(
            classify_action_alignment(
                base,
                {
                    "lora": {3: 1.1, 4: 2.1},
                    "action_expert": {3: 1.2, 4: 2.2},
                },
            )["status"],
            "fixed_flow_query_surrogate_misaligned",
        )
        self.assertEqual(
            classify_action_alignment(
                base,
                {
                    "lora": {3: 0.9, 4: 1.9},
                    "action_expert": {3: 0.8, 4: 1.8},
                },
            )["status"],
            "generated_action_error_improves_without_closed_loop_conversion",
        )
        self.assertEqual(
            classify_action_alignment(
                base,
                {
                    "lora": {3: 0.9, 4: 2.1},
                    "action_expert": {3: 1.2, 4: 1.8},
                },
            )["status"],
            "aggregate_query_hides_action_error_heterogeneity",
        )

    def test_multi_seed_classification_requires_mean_and_three_of_four_worse(self) -> None:
        base = {3: [1.0] * 4, 4: [2.0] * 4}
        robust = classify_multi_seed_action_alignment(
            base,
            {
                "lora": {3: [1.1, 1.2, 1.1, 0.9], 4: [2.1, 2.2, 2.1, 1.9]},
                "action_expert": {3: [1.01, 1.02, 1.03, 0.99], 4: [2.01, 2.02, 2.03, 1.99]},
            },
            minimum_consistent_worsening_draws=3,
        )
        mixed = classify_multi_seed_action_alignment(
            base,
            {
                "lora": {3: [1.2, 1.2, 0.8, 0.8], 4: [2.2, 2.2, 1.8, 1.8]},
                "action_expert": {3: [1.1, 1.1, 0.9, 0.9], 4: [2.1, 2.1, 1.9, 1.9]},
            },
            minimum_consistent_worsening_draws=3,
        )

        self.assertEqual(robust["status"], "multi_seed_flow_query_surrogate_misaligned")
        self.assertEqual(mixed["status"], "inference_sampling_variance_obscures_alignment")

    def test_launcher_dry_run_uses_two_task_parallel_ranks(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_query_action_alignment.sh"),
                "--gpus=4,5",
                "--output-dir=/tmp/ember-query-action-alignment-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--nproc-per-node=2", completed.stdout)
        self.assertIn("gate_zero_query_action_alignment", completed.stdout)


if __name__ == "__main__":
    unittest.main()
