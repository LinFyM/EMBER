from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.evaluation_identity import (  # noqa: E402
    IdentityProbeError,
    canonical_tree_summary,
    compare_trees,
    load_probe_spec,
    policy_recovery_allowed,
)


class EvaluationIdentityTest(unittest.TestCase):
    def test_tree_digest_is_mapping_order_invariant(self) -> None:
        first = {
            "state": np.array([1.0, 2.0], dtype=np.float32),
            "image": np.arange(6, dtype=np.uint8).reshape(2, 3),
        }
        second = {"image": first["image"].copy(), "state": first["state"].copy()}

        self.assertEqual(
            canonical_tree_summary(first)["sha256"],
            canonical_tree_summary(second)["sha256"],
        )

    def test_tree_digest_binds_shape_and_dtype(self) -> None:
        vector = canonical_tree_summary(np.array([1, 2], dtype=np.int32))
        matrix = canonical_tree_summary(np.array([[1, 2]], dtype=np.int32))
        wider = canonical_tree_summary(np.array([1, 2], dtype=np.int64))

        self.assertNotEqual(vector["sha256"], matrix["sha256"])
        self.assertNotEqual(vector["sha256"], wider["sha256"])

    def test_tree_comparison_reports_numeric_divergence(self) -> None:
        left = {"x": np.array([1.0, 2.0], dtype=np.float32)}
        right = {"x": np.array([1.0, 2.0 + 5e-7], dtype=np.float32)}

        exact = compare_trees(left, right, atol=0.0, rtol=0.0)
        tolerant = compare_trees(left, right, atol=1e-6, rtol=1e-6)

        self.assertFalse(exact["exact"])
        self.assertFalse(exact["within_tolerance"])
        self.assertGreater(exact["max_abs"], 0.0)
        self.assertTrue(tolerant["within_tolerance"])
        self.assertEqual(exact["mismatched_paths"], ["x"])
        self.assertEqual(exact["leaf_differences"]["x"]["unequal_count"], 1)
        self.assertEqual(exact["mismatch_domains"], ["other"])

    def test_checked_in_probe_spec_is_bounded_and_overlap_only(self) -> None:
        spec = load_probe_spec(ROOT / "configs" / "gate_minus1_identity.toml")

        self.assertEqual(spec["surface"], "official_overlap_mechanics_only")
        self.assertEqual(spec["policy_batch_sizes"][-1], 112)
        self.assertEqual({item["mode"] for item in spec["env_conditions"]}, {"sync", "async"})
        self.assertEqual(len(spec["env_conditions"]), 6)
        self.assertEqual(spec["fixed_steps"], 5)

    def test_probe_spec_rejects_held_surface(self) -> None:
        source = (ROOT / "configs" / "gate_minus1_identity.toml").read_text()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "probe.toml"
            path.write_text(
                source.replace(
                    'surface = "official_overlap_mechanics_only"',
                    'surface = "held"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(IdentityProbeError, "overlap"):
                load_probe_spec(path)

    def test_recovery_spec_authorizes_only_pixel_only_mechanics_failure(self) -> None:
        strict_spec = load_probe_spec(ROOT / "configs" / "gate_minus1_identity.toml")
        spec = load_probe_spec(ROOT / "configs" / "gate_minus1_identity_recovery.toml")
        pixel_only = [
            {
                "reset_observation": {"mismatch_domains": ["pixels"]},
                "fixed_trajectory": {"mismatch_domains": ["pixels"]},
            }
        ]
        state_divergence = [
            {
                "reset_observation": {"mismatch_domains": ["pixels", "state"]},
                "fixed_trajectory": {"mismatch_domains": ["pixels"]},
            }
        ]

        self.assertTrue(
            policy_recovery_allowed(spec, "reset_observation_mismatch", pixel_only)
        )
        self.assertFalse(
            policy_recovery_allowed(spec, "reset_observation_mismatch", state_divergence)
        )
        self.assertFalse(policy_recovery_allowed(spec, None, pixel_only))
        self.assertFalse(
            policy_recovery_allowed(strict_spec, "reset_observation_mismatch", pixel_only)
        )
        strict_spec.pop("mechanics_failure_policy")
        spec.pop("mechanics_failure_policy")
        self.assertEqual(strict_spec, spec)

    def test_shell_entrypoint_dry_run_is_offline_and_single_gpu(self) -> None:
        environment = os.environ.copy()
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / "run_evaluation_identity_probe.sh"),
                "--gpu=5",
                "--output-dir=/tmp/ember-identity-probe-test",
                "--mechanics-only",
                "--dry-run",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=5", result.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", result.stdout)
        self.assertIn("gate_minus1_identity.toml", result.stdout)
        self.assertIn("ember.evaluation_identity", result.stdout)
        self.assertIn("--mechanics-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
