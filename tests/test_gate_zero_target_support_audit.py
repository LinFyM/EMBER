from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_oracle_session import resolve_lora_variant_spec  # noqa: E402
from ember.gate_zero_support.contract import (  # noqa: E402
    GateZeroTargetSupportContractError,
    load_target_support_audit_spec,
    load_target_support_rank16_spec,
)


class GateZeroTargetSupportAuditContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audit_path = ROOT / "configs" / "gate_zero_target_support_audit.toml"
        self.rank16_path = ROOT / "configs" / "gate_zero_target_support_rank16.toml"
        self.gate_zero_path = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0_path = ROOT / "configs" / "phase0.toml"
        self.competence_path = ROOT / "configs" / "gate_zero_source_competence.toml"
        self.prior_execution_path = ROOT / "configs" / "gate_zero_oracle_execution.toml"

    def load(self):
        return load_target_support_audit_spec(
            self.audit_path,
            gate_zero_path=self.gate_zero_path,
            phase0_path=self.phase0_path,
            competence_path=self.competence_path,
            prior_execution_path=self.prior_execution_path,
        )

    def test_checked_in_audit_is_bounded_held_zero_and_compares_required_supports(self) -> None:
        spec = self.load()

        self.assertEqual(
            spec["variants"],
            ["last_two_qv_r8", "all_expert_qv_r8", "official_default_r8"],
        )
        self.assertEqual(spec["fit"]["optimizer_steps"], 750)
        self.assertEqual(
            spec["fit"]["candidate_steps"], [0, 25, 50, 100, 150, 250, 500, 750]
        )
        self.assertEqual(
            spec["fit"]["last_two_qv_r8"]["learning_rate"], 0.0001
        )
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])
        self.assertFalse(spec["authority"]["locked_report_numeric_reuse_for_selection"])

        last_two = set(spec["fit"]["last_two_qv_r8"]["target_modules"])
        all_expert = set(spec["fit"]["all_expert_qv_r8"]["target_modules"])
        official = set(spec["fit"]["official_default_r8"]["target_modules"])
        self.assertEqual(len(last_two), 4)
        self.assertEqual(len(all_expert), 32)
        self.assertEqual(len(official), 37)
        self.assertLess(last_two, all_expert)
        self.assertLess(all_expert, official)
        self.assertEqual(
            official - all_expert,
            {
                "model.state_proj",
                "model.action_in_proj",
                "model.action_out_proj",
                "model.action_time_mlp_in",
                "model.action_time_mlp_out",
            },
        )
        self.assertEqual(
            [
                spec["fit"][name]["expected_trainable_parameters"]
                for name in spec["variants"]
            ],
            [40320, 322560, 371328],
        )
        self.assertEqual(spec["screening_rollout"]["init_state_indices"], list(range(24, 32)))
        self.assertEqual(spec["confirmation_rollout"]["init_state_indices"], list(range(32, 40)))
        self.assertTrue(spec["rank_escalation"]["conditional_only"])
        self.assertEqual(spec["rank_escalation"]["rank"], 16)
        self.assertEqual(spec["rank_escalation"]["maximum_additional_supports"], 1)

    def test_prior_frozen_failure_hash_drift_fails_closed(self) -> None:
        changed = self.audit_path.read_text(encoding="utf-8").replace(
            "b7fcfc6227ba7fd6fc2e9ad21b2e55978b54d668476c9c520e216536739e9d91",
            "0" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(
                GateZeroTargetSupportContractError, "prior locked report SHA256"
            ):
                load_target_support_audit_spec(
                    path,
                    gate_zero_path=self.gate_zero_path,
                    phase0_path=self.phase0_path,
                    competence_path=self.competence_path,
                    prior_execution_path=self.prior_execution_path,
                )

    def test_new_named_variants_and_legacy_lora_resolve_through_one_model_path(self) -> None:
        spec = self.load()
        parent = {
            "oracle": {
                "target_modules": ["legacy.q_proj"],
                "init_lora_weights": True,
            },
            "authority": {"model_revision": "a" * 40},
        }

        legacy = resolve_lora_variant_spec(
            parent=parent,
            variant="lora",
            variant_spec={"rank": 8, "alpha": 8, "dropout": 0.0},
        )
        audited = resolve_lora_variant_spec(
            parent=parent,
            variant="all_expert_qv_r8",
            variant_spec=spec["fit"]["all_expert_qv_r8"],
        )

        self.assertEqual(legacy["target_modules"], ["legacy.q_proj"])
        self.assertEqual(
            audited["target_modules"],
            spec["fit"]["all_expert_qv_r8"]["target_modules"],
        )
        self.assertEqual(audited["rank"], 8)
        self.assertEqual(audited["alpha"], 8)
        self.assertEqual(audited["dropout"], 0.0)

    def test_canonical_launcher_accepts_a_declared_audit_variant(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_fit.sh"),
                f"--config={self.audit_path}",
                "--variant=all_expert_qv_r8",
                "--task-id=3",
                "--gpu=4",
                "--output-dir=/tmp/ember-gate0-support-audit-test",
                "--latest-link=/tmp/ember-gate0-support-audit-latest-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_oracle_fit", completed.stdout)
        self.assertIn(str(self.audit_path), completed.stdout)
        self.assertIn("--variant all_expert_qv_r8", completed.stdout)

    def test_rank16_contract_is_the_single_hash_bound_conditional_scope(self) -> None:
        spec = load_target_support_rank16_spec(
            self.rank16_path,
            gate_zero_path=self.gate_zero_path,
            phase0_path=self.phase0_path,
            competence_path=self.competence_path,
            prior_execution_path=self.prior_execution_path,
            rank8_audit_path=self.audit_path,
        )
        rank8 = self.load()

        self.assertEqual(spec["variants"], ["official_default_r16"])
        candidate = spec["fit"]["official_default_r16"]
        self.assertEqual(candidate["rank"], 16)
        self.assertEqual(candidate["alpha"], 16)
        self.assertEqual(candidate["dropout"], 0.0)
        self.assertEqual(candidate["expected_trainable_parameters"], 742656)
        self.assertEqual(
            candidate["target_modules"],
            rank8["fit"]["official_default_r8"]["target_modules"],
        )
        self.assertEqual(spec["screening_stage"], "rank16")
        self.assertEqual(
            spec["screening_rollout"]["init_state_indices"], list(range(32, 40))
        )
        self.assertEqual(
            spec["confirmation_rollout"]["init_state_indices"], list(range(40, 48))
        )
        self.assertEqual(
            spec["authority"]["rank8_screening_result_sha256"],
            "0df3acb8d3fd5f94507921298940281c7430eedc359869d3918a0f2c012c6efb",
        )
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])
        self.assertTrue(spec["rank_escalation"]["no_further_support_or_rank_search"])

    def test_canonical_fitter_accepts_only_the_declared_rank16_variant(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_fit.sh"),
                f"--config={self.rank16_path}",
                "--variant=official_default_r16",
                "--task-id=3",
                "--gpu=4",
                "--output-dir=/tmp/ember-gate0-rank16-fit-test",
                "--latest-link=/tmp/ember-gate0-rank16-fit-latest-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_oracle_fit", completed.stdout)
        self.assertIn("--variant official_default_r16", completed.stdout)


if __name__ == "__main__":
    unittest.main()
