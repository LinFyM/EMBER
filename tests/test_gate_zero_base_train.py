from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_base_train import (  # noqa: E402
    GateZeroBaseTrainError,
    assert_exact_resume_equivalence,
    build_source_base_checkpoint_metadata,
    canonical_state_sha256,
    require_base_fit_authorization,
    should_log_training_step,
)
from ember.gate_zero_contract import load_gate_zero_contract  # noqa: E402


class GateZeroBaseTrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.spec = load_gate_zero_contract(self.config, self.phase0)

    def test_canonical_state_digest_is_tensor_exact_and_order_stable(self) -> None:
        left = {"b": [torch.tensor([1, 2]), 3], "a": torch.tensor(1.0)}
        right = {"a": torch.tensor(1.0), "b": [torch.tensor([1, 2]), 3]}

        self.assertEqual(canonical_state_sha256(left), canonical_state_sha256(right))
        right["b"][0][1] = 4
        self.assertNotEqual(canonical_state_sha256(left), canonical_state_sha256(right))

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        parameter.square().backward()
        optimizer.step()
        self.assertEqual(len(canonical_state_sha256(optimizer.state_dict())), 64)

    def test_resume_equivalence_fails_closed_on_any_state_surface(self) -> None:
        authority = {
            "completed_step": 2,
            "model_state_sha256": "a" * 64,
            "optimizer_state_sha256": "b" * 64,
            "scheduler_state_sha256": "c" * 64,
            "rng_state_sha256": "d" * 64,
            "next_raw_batch_sha256": "e" * 64,
            "next_row_keys_sha256": "f" * 64,
        }

        comparison = assert_exact_resume_equivalence(authority, dict(authority))

        self.assertTrue(comparison["all_exact"])
        changed = dict(authority)
        changed["rng_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(GateZeroBaseTrainError, "rng_state_sha256"):
            assert_exact_resume_equivalence(authority, changed)

    def test_frozen_resume_authority_allows_formal_fit(self) -> None:
        require_base_fit_authorization(self.spec, mode="resume-probe")
        require_base_fit_authorization(self.spec, mode="train")

    def test_checkpoint_metadata_binds_topology_and_contract_hashes(self) -> None:
        metadata = build_source_base_checkpoint_metadata(
            self.spec,
            config_path=self.config,
            phase0_path=self.phase0,
            completed_step=1,
        )

        self.assertEqual(metadata["checkpoint_role"], "source_base_training_recovery")
        self.assertEqual(metadata["topology"]["micro_batch_size"], 64)
        self.assertEqual(metadata["topology"]["gradient_accumulation_steps"], 1)
        self.assertEqual(metadata["sampler"]["next_optimizer_step"], 1)
        self.assertEqual(len(metadata["authorities"]["gate_zero_contract_sha256"]), 64)
        self.assertEqual(len(metadata["authorities"]["phase0_contract_sha256"]), 64)
        self.assertEqual(len(metadata["authorities"]["implementation_files_sha256"]), 6)

    def test_training_logging_honors_frozen_cadence_and_boundaries(self) -> None:
        selected = [
            step
            for step in range(1, 26)
            if should_log_training_step(step, target_step=25, every=10)
        ]

        self.assertEqual(selected, [1, 10, 20, 25])

    def test_shell_dry_run_exposes_one_canonical_resume_probe_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_base_train.sh"),
                "--mode=resume-probe",
                "--gpu=4",
                "--output-dir=/tmp/ember-gate-zero-resume-probe",
                "--latest-link=/tmp/ember-gate-zero-resume-probe-latest",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=4", completed.stdout)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", completed.stdout)
        self.assertIn("-m ember.gate_zero_base_train", completed.stdout)
        self.assertIn("--mode resume-probe", completed.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
