from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_support.contract import load_target_support_screen_spec  # noqa: E402
from ember.gate_zero_support.mature_headroom import (  # noqa: E402
    GateZeroMatureLoraHeadroomContractError,
    decide_mature_lora_headroom,
)


class GateZeroMatureLoraHeadroomTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_mature_lora_headroom_screen.toml"
        self.fit_config = ROOT / "configs" / "gate_zero_mature_lora_lr_recovery.toml"
        self.fit_ladder = (
            ROOT / "configs" / "gate_zero_mature_lora_lr_recovery_ladder.toml"
        )
        self.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"
        self.variant = "mature_official_default_r32_lr25e6_recovery"

    def load(self):
        return load_target_support_screen_spec(
            self.config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
            prior_execution_path=ROOT / "configs" / "gate_zero_oracle_execution.toml",
        )

    def test_contract_binds_step1000_and_ceiling_safe_decision(self) -> None:
        spec = self.load()

        self.assertEqual(spec["screening_stage"], "mature_lora_headroom_control")
        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["variants"], [self.variant])
        self.assertEqual(spec["staged_selection"]["selected_step"], 1_000)
        self.assertEqual(
            spec["authority"]["fit_contract_sha256"],
            hashlib.sha256(self.fit_config.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            spec["authority"]["fit_ladder_sha256"],
            hashlib.sha256(self.fit_ladder.read_bytes()).hexdigest(),
        )
        self.assertEqual(spec["decision"]["maintenance_task_id"], 3)
        self.assertEqual(spec["decision"]["improvement_task_id"], 4)
        self.assertEqual(spec["decision"]["minimum_improvement_net_wins"], 2)
        self.assertEqual(spec["decision"]["minimum_maintenance_net_wins"], 0)
        self.assertEqual(spec["decision"]["minimum_each_query_reduction_fraction"], 0.02)
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])

    def test_contract_drift_fails_closed(self) -> None:
        changed = self.config.read_text(encoding="utf-8").replace(
            "minimum_improvement_net_wins = 2",
            "minimum_improvement_net_wins = 1",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "headroom.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroMatureLoraHeadroomContractError):
                load_target_support_screen_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    prior_execution_path=ROOT
                    / "configs"
                    / "gate_zero_oracle_execution.toml",
                )

    @staticmethod
    def _arm(task_id: int, condition: str, successes: list[bool]) -> dict:
        return {
            "task_id": task_id,
            "condition": condition,
            "successes": successes,
            "mechanics_valid": True,
            "official_rollout_init_state_indices": list(range(40, 48)),
            "seeds": list(range(5800, 5808)),
        }

    def _grant(self) -> dict:
        return {
            "fit_evidence": {
                f"{self.variant}:task3": {
                    "selected_query_metrics": {
                        "base_query_flow_mse": 1.0,
                        "query_flow_mse": 0.94,
                        "action_drift_proxy": 0.011,
                    }
                },
                f"{self.variant}:task4": {
                    "selected_query_metrics": {
                        "base_query_flow_mse": 1.0,
                        "query_flow_mse": 0.95,
                        "action_drift_proxy": 0.009,
                    }
                },
            }
        }

    def _thresholds(self) -> dict:
        return {
            "maintenance_task_id": 3,
            "improvement_task_id": 4,
            "minimum_improvement_headroom_failures": 2,
            "minimum_improvement_net_wins": 2,
            "minimum_maintenance_net_wins": 0,
            "minimum_aggregate_net_wins": 2,
            "minimum_each_query_reduction_fraction": 0.02,
            "maximum_each_selection_drift_proxy": 0.02,
        }

    def test_ceiling_maintenance_plus_two_paired_wins_passes(self) -> None:
        base3 = [True] * 8
        own3 = [True] * 8
        base4 = [True, True, True, True, True, False, False, False]
        own4 = [True, True, True, True, True, True, True, False]
        decision = decide_mature_lora_headroom(
            arms=[
                self._arm(3, "frozen_base", base3),
                self._arm(3, self.variant, own3),
                self._arm(4, "frozen_base", base4),
                self._arm(4, self.variant, own4),
            ],
            grant=self._grant(),
            variant=self.variant,
            parameter_count=1_485_312,
            thresholds=self._thresholds(),
        )

        self.assertEqual(decision["status"], "mature_lora_headroom_control_passed")
        self.assertEqual(decision["selected_variant"], self.variant)
        self.assertTrue(decision["gate_zero_authorized"])
        self.assertTrue(decision["writer_authorized"])
        self.assertTrue(decision["final_writer_target_contract_sealed"])
        self.assertEqual(
            decision["candidates"][0]["task_metrics"]["4"]["paired_net_wins"], 2
        )

    def test_task3_harm_fails_even_when_task4_improves(self) -> None:
        decision = decide_mature_lora_headroom(
            arms=[
                self._arm(3, "frozen_base", [True] * 8),
                self._arm(3, self.variant, [True] * 7 + [False]),
                self._arm(4, "frozen_base", [True] * 5 + [False] * 3),
                self._arm(4, self.variant, [True] * 7 + [False]),
            ],
            grant=self._grant(),
            variant=self.variant,
            parameter_count=1_485_312,
            thresholds=self._thresholds(),
        )

        self.assertEqual(
            decision["status"],
            "mature_lora_headroom_control_failed_gate_recovery_required",
        )
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_absent_task4_headroom_is_distinct_and_cannot_pass(self) -> None:
        decision = decide_mature_lora_headroom(
            arms=[
                self._arm(3, "frozen_base", [True] * 8),
                self._arm(3, self.variant, [True] * 8),
                self._arm(4, "frozen_base", [True] * 7 + [False]),
                self._arm(4, self.variant, [True] * 8),
            ],
            grant=self._grant(),
            variant=self.variant,
            parameter_count=1_485_312,
            thresholds=self._thresholds(),
        )

        self.assertEqual(
            decision["status"],
            "mature_lora_headroom_absent_source_recovery_required",
        )
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_canonical_launcher_accepts_staged_candidate_outputs(self) -> None:
        script = ROOT / "scripts" / "run_gate_zero_target_support_screen.sh"
        source = script.read_text(encoding="utf-8")
        self.assertNotIn('fit_root/$name/selected', source)
        self.assertIn('fit output is missing', source)

        completed = subprocess.run(
            [
                str(script),
                f"--config={self.config}",
                "--gpus=4,5",
                "--fit-root=/tmp/ember-mature-headroom-fit-root",
                "--screening-freeze-dir=/tmp/ember-mature-headroom-freeze",
                "--output-dir=/tmp/ember-mature-headroom-screen",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--nproc-per-node=2", completed.stdout)
        self.assertIn("ember.gate_zero_support.screen_runtime", completed.stdout)


if __name__ == "__main__":
    unittest.main()
