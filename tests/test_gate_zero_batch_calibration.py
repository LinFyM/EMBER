from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_batch_calibration import (  # noqa: E402
    GateZeroBatchCalibrationError,
    assert_matched_candidate_records,
    gradient_accumulation_steps,
    select_calibration_candidate,
)


class GateZeroBatchCalibrationTest(unittest.TestCase):
    def test_accumulation_preserves_effective_batch(self) -> None:
        self.assertEqual(gradient_accumulation_steps(64, 8), 8)
        self.assertEqual(gradient_accumulation_steps(64, 16), 4)
        self.assertEqual(gradient_accumulation_steps(64, 32), 2)
        self.assertEqual(gradient_accumulation_steps(64, 64), 1)
        with self.assertRaisesRegex(GateZeroBatchCalibrationError, "divide"):
            gradient_accumulation_steps(64, 24)

    def test_selection_uses_fastest_completed_candidate_with_headroom(self) -> None:
        records = [
            {
                "micro_batch_size": 8,
                "status": "completed",
                "samples_per_second": 10.0,
                "minimum_free_memory_mib": 50000,
            },
            {
                "micro_batch_size": 16,
                "status": "completed",
                "samples_per_second": 20.0,
                "minimum_free_memory_mib": 40000,
            },
            {
                "micro_batch_size": 32,
                "status": "completed",
                "samples_per_second": 30.0,
                "minimum_free_memory_mib": 9000,
            },
            {
                "micro_batch_size": 64,
                "status": "oom",
                "samples_per_second": None,
                "minimum_free_memory_mib": None,
            },
        ]

        selected = select_calibration_candidate(records, minimum_free_memory_mib=10240)

        self.assertEqual(selected["micro_batch_size"], 16)
        self.assertEqual(selected["gradient_accumulation_steps"], 4)

    def test_selection_fails_closed_without_safe_candidate(self) -> None:
        with self.assertRaisesRegex(GateZeroBatchCalibrationError, "safe candidate"):
            select_calibration_candidate(
                [
                    {
                        "micro_batch_size": 8,
                        "status": "completed",
                        "samples_per_second": 1.0,
                        "minimum_free_memory_mib": 9000,
                    }
                ],
                minimum_free_memory_mib=10240,
            )

    def test_matched_candidate_check_requires_identical_effective_batches(self) -> None:
        matched = [
            {
                "micro_batch_size": 8,
                "status": "completed",
                "matched_initial_trainable_state": True,
                "fixed_flow_seed": 31,
                "optimizer_step_row_keys_sha256": ["a" * 64, "b" * 64],
            },
            {
                "micro_batch_size": 16,
                "status": "completed",
                "matched_initial_trainable_state": True,
                "fixed_flow_seed": 31,
                "optimizer_step_row_keys_sha256": ["a" * 64, "b" * 64],
            },
        ]

        authority = assert_matched_candidate_records(matched)

        self.assertEqual(authority["fixed_flow_seed"], 31)
        self.assertEqual(authority["optimizer_step_row_keys_sha256"], ["a" * 64, "b" * 64])
        matched[1]["optimizer_step_row_keys_sha256"][1] = "c" * 64
        with self.assertRaisesRegex(GateZeroBatchCalibrationError, "effective-batch draws"):
            assert_matched_candidate_records(matched)

    def test_matched_candidate_check_rejects_unrestored_state(self) -> None:
        with self.assertRaisesRegex(GateZeroBatchCalibrationError, "initial trainable state"):
            assert_matched_candidate_records(
                [
                    {
                        "micro_batch_size": 8,
                        "status": "completed",
                        "matched_initial_trainable_state": False,
                        "fixed_flow_seed": 31,
                        "optimizer_step_row_keys_sha256": ["a" * 64],
                    }
                ]
            )

    def test_shell_dry_run_has_one_canonical_offline_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_batch_calibration.sh"),
                "--gpu=4",
                "--output-dir=/tmp/ember-gate-zero-calibration",
                "--latest-link=/tmp/ember-gate-zero-calibration-latest",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=4", completed.stdout)
        self.assertIn("-m ember.gate_zero_batch_calibration", completed.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
