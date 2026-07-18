from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from ember.gate_zero_oracle_artifacts import (
    publish_selected_artifact,
    save_candidate_artifact,
    sha256_file,
    validate_selected_artifact,
    write_output_checksums,
)
from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST
from ember.gate_zero_oracle_report import (
    assigned_report_arms,
    canonical_report_shards,
    create_selection_freeze_grant,
    decide_gate_zero_report,
    validate_selection_freeze_grant,
)
from ember.gate_zero_oracle_report_runtime import (
    checkpoint_manifest_path,
    report_state_authority,
    validate_report_reset_identity,
)


ROOT = Path(__file__).resolve().parents[1]


class GateZeroOracleReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.execution = ROOT / "configs" / "gate_zero_oracle_execution.toml"
        self.parent = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"

    def _fit_output(self, root: Path, *, variant: str, task_id: int) -> Path:
        authorities = {
            "execution_contract_sha256": sha256_file(self.execution),
            "gate_zero_contract_sha256": sha256_file(self.parent),
            "phase0_contract_sha256": sha256_file(self.phase0),
            "source_competence_contract_sha256": sha256_file(self.competence),
        }
        output = root / f"{variant}_task{task_id}"
        output.mkdir()
        candidate = save_candidate_artifact(
            output,
            variant=variant,
            task_id=task_id,
            step=500,
            trainable_state={"weight": torch.tensor([float(task_id)])},
            metrics={
                "step": 500,
                "query_flow_mse": 0.7,
                "base_query_flow_mse": 1.0,
                "action_drift_proxy": 0.01,
            },
            authorities=authorities,
        )
        selected_dir = publish_selected_artifact(output, candidate)
        selected = validate_selected_artifact(selected_dir)
        result = {
            "schema_version": 1,
            "status": "oracle_fit_selection_complete_pending_global_report_grant",
            "variant": variant,
            "task_id": task_id,
            "authorities": authorities,
            "selection": {
                "selected_step": selected["selected_step"],
                "selected_trainable_state_sha256": selected["trainable_state_sha256"],
                "selected_manifest_sha256": sha256_file(
                    selected_dir / "selected_manifest.json"
                ),
                "locked_report_accessed": False,
            },
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "final_writer_target_contract_sealed": False,
        }
        (output / "fit_selection_result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_output_checksums(output)
        return output

    def test_report_shards_are_predeclared_and_partition_without_duplication(self) -> None:
        shards = canonical_report_shards()

        self.assertEqual(
            shards,
            [
                [(3, "frozen_base"), (3, "own_adapter")],
                [(3, "swapped_adapter"), (3, "partial_upper_bound")],
                [(4, "frozen_base"), (4, "own_adapter")],
                [(4, "swapped_adapter"), (4, "partial_upper_bound")],
            ],
        )
        world4 = [assigned_report_arms(rank=rank, world_size=4) for rank in range(4)]
        self.assertEqual(world4, shards)
        self.assertEqual(
            sorted(arm for rank_arms in world4 for arm in rank_arms),
            sorted(arm for shard in shards for arm in shard),
        )

    def test_selection_freeze_validates_all_four_fit_outputs_before_authorizing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {
                (variant, task_id): self._fit_output(
                    root, variant=variant, task_id=task_id
                )
                for variant in ("lora", "partial_upper_bound")
                for task_id in (3, 4)
            }
            grant_path = root / "freeze" / "selection_freeze_grant.json"

            grant = create_selection_freeze_grant(
                execution_path=self.execution,
                parent_path=self.parent,
                phase0_path=self.phase0,
                competence_path=self.competence,
                fit_outputs=outputs,
                grant_path=grant_path,
            )

            self.assertEqual(
                grant["status"], "oracle_selection_frozen_before_report_access"
            )
            self.assertTrue(grant["report_access_authorized"])
            self.assertEqual(set(grant["selected_adapter_sha256_by_task"]), {"3", "4"})
            self.assertEqual(
                set(grant["selected_capacity_upper_bound_sha256_by_task"]), {"3", "4"}
            )
            self.assertFalse(grant["gate_zero_authorized"])
            self.assertFalse(grant["writer_authorized"])
            self.assertEqual(
                json.loads(grant_path.read_text(encoding="utf-8")), grant
            )
            validated = validate_selection_freeze_grant(
                grant_path=grant_path,
                execution_path=self.execution,
                parent_path=self.parent,
                phase0_path=self.phase0,
                competence_path=self.competence,
                fit_outputs=outputs,
            )
            self.assertEqual(validated, grant)

            outputs.pop(("partial_upper_bound", 4))
            with self.assertRaisesRegex(Exception, "four frozen fit outputs"):
                create_selection_freeze_grant(
                    execution_path=self.execution,
                    parent_path=self.parent,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    fit_outputs=outputs,
                    grant_path=root / "other" / "selection_freeze_grant.json",
                )

    @staticmethod
    def _arms(*, own_successes: int, partial_successes: int) -> list[dict[str, object]]:
        arms = []
        for task_id in (3, 4):
            for condition, successes, flow in (
                ("frozen_base", 2, 1.0),
                ("own_adapter", own_successes, 0.7 if own_successes > 2 else 1.0),
                ("swapped_adapter", 1, 1.1),
                ("partial_upper_bound", partial_successes, 0.5),
            ):
                arms.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "mechanics_valid": True,
                        "successes": [True] * successes + [False] * (8 - successes),
                        "offline_flow_mse": flow,
                        "base_offline_flow_mse": 1.0,
                        "offline_sample_count": 128,
                        "offline_row_keys_sha256": str(task_id) * 64,
                        "official_rollout_init_state_indices": list(range(16, 24)),
                        "seeds": list(range(5400, 5408)),
                    }
                )
        return arms

    def test_gate_zero_decision_requires_primary_lora_not_partial_upper_bound(self) -> None:
        passing_arms = self._arms(own_successes=4, partial_successes=6)
        passing_arms[1]["base_offline_flow_mse"] = 1.00005
        passing = decide_gate_zero_report(
            arms=passing_arms,
            selected_lora_drift={3: 0.01, 4: 0.015},
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 0.70,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
            },
        )

        self.assertEqual(passing["status"], "gate_zero_pilot_passed")
        self.assertTrue(passing["gate_zero_pilot_passed"])
        self.assertFalse(passing["writer_authorized"])

        passing_arms[1]["base_offline_flow_mse"] = 1.001
        with self.assertRaisesRegex(Exception, "base loss differs"):
            decide_gate_zero_report(
                arms=passing_arms,
                selected_lora_drift={3: 0.01, 4: 0.015},
                thresholds={
                    "median_success_gain_pp_min": 15.0,
                    "median_locked_action_loss_reduction_fraction_min": 0.20,
                    "positive_task_fraction_min": 0.70,
                    "median_selection_drift_proxy_max": 0.02,
                    "two_task_positive_count_required": 2,
                },
            )

        partial_only = decide_gate_zero_report(
            arms=self._arms(own_successes=2, partial_successes=6),
            selected_lora_drift={3: 0.01, 4: 0.015},
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 0.70,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
            },
        )

        self.assertEqual(partial_only["status"], "gate_zero_pilot_failed")
        self.assertEqual(
            partial_only["failure_class"],
            "primary_lora_contract_too_narrow_trigger_bounded_recovery",
        )
        self.assertFalse(partial_only["gate_zero_pilot_passed"])

    def test_report_arm_state_mapping_and_two_reset_identity_are_exact(self) -> None:
        self.assertEqual(
            checkpoint_manifest_path(Path("/tmp/checkpoint")),
            Path("/tmp/checkpoint") / CHECKPOINT_MANIFEST,
        )
        self.assertEqual(report_state_authority(3, "frozen_base"), (None, None))
        self.assertEqual(report_state_authority(3, "own_adapter"), ("lora", 3))
        self.assertEqual(report_state_authority(3, "swapped_adapter"), ("lora", 4))
        self.assertEqual(
            report_state_authority(3, "partial_upper_bound"),
            ("partial_upper_bound", 3),
        )
        events = [
            {
                "before": list(range(8)),
                "after": list(range(8, 16)),
                "seeds": list(range(5392, 5400)),
            },
            {
                "before": list(range(8, 16)),
                "after": list(range(16, 24)),
                "seeds": list(range(5400, 5408)),
            },
        ]
        self.assertTrue(
            validate_report_reset_identity(
                events,
                batch_size=8,
                warmup_seed_start=5392,
                report_seed_start=5400,
                expected_report_init_states=list(range(16, 24)),
            )
        )
        events[1]["after"][-1] = 99
        self.assertFalse(
            validate_report_reset_identity(
                events,
                batch_size=8,
                warmup_seed_start=5392,
                report_seed_start=5400,
                expected_report_init_states=list(range(16, 24)),
            )
        )

    def test_locked_report_launcher_dry_run_exposes_one_parallel_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_report.sh"),
                "--gpus=4,5,6,7",
                "--selection-freeze-dir=/tmp/ember-gate0-freeze-test",
                "--output-dir=/tmp/ember-gate0-report-test",
                "--latest-link=/tmp/ember-gate0-report-latest-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_oracle_report freeze", completed.stdout)
        self.assertIn("-m torch.distributed.run", completed.stdout)
        self.assertIn("-m ember.gate_zero_oracle_report_runtime", completed.stdout)
        self.assertIn("--nproc-per-node=4", completed.stdout)


if __name__ == "__main__":
    unittest.main()
