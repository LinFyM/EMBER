from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_oracle_artifacts import (  # noqa: E402
    publish_selected_artifact,
    save_candidate_artifact,
    sha256_file,
    validate_selected_artifact,
    write_output_checksums,
)
from ember.gate_zero_support.screen import (  # noqa: E402
    assigned_support_screening_arms,
    canonical_support_screening_shards,
    create_support_screening_grant,
    decide_support_screening,
    validate_support_screening_grant,
)
from ember.gate_zero_support.screen_runtime import support_state_authority  # noqa: E402


class GateZeroTargetSupportScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_target_support_audit.toml"
        self.parent = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"
        self.variants = [
            "last_two_qv_r8",
            "all_expert_qv_r8",
            "official_default_r8",
        ]
        self.parameter_counts = {
            "last_two_qv_r8": 40320,
            "all_expert_qv_r8": 322560,
            "official_default_r8": 371328,
        }

    def _fit_output(
        self,
        root: Path,
        *,
        variant: str,
        task_id: int,
        reduction: float,
        config_path: Path | None = None,
        parameter_count: int | None = None,
    ) -> Path:
        config_path = self.config if config_path is None else config_path
        parameter_count = (
            self.parameter_counts[variant]
            if parameter_count is None
            else parameter_count
        )
        authorities = {
            "execution_contract_sha256": sha256_file(config_path),
            "gate_zero_contract_sha256": sha256_file(self.parent),
            "phase0_contract_sha256": sha256_file(self.phase0),
            "source_competence_contract_sha256": sha256_file(self.competence),
            "validation_numeric_access": False,
            "held_numeric_access": False,
        }
        output = root / f"{variant}_task{task_id}"
        output.mkdir()
        candidate = save_candidate_artifact(
            output,
            variant=variant,
            task_id=task_id,
            step=100,
            trainable_state={
                "weight": torch.zeros(parameter_count, dtype=torch.bfloat16)
            },
            metrics={
                "step": 100,
                "query_flow_mse": 1.0 - reduction,
                "base_query_flow_mse": 1.0,
                "query_sample_count": 128,
                "query_row_keys_sha256": str(task_id) * 64,
                "anchor_count": 48,
                "anchor_row_keys_sha256": str(task_id + 1) * 64,
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
            "pilot_scope": (
                "source_only_gate_zero_target_support_audit_"
                "not_final_writer_target_support"
            ),
            "authorities": authorities,
            "selection": {
                "selected_step": selected["selected_step"],
                "selected_trainable_state_sha256": selected[
                    "trainable_state_sha256"
                ],
                "selected_manifest_sha256": sha256_file(
                    selected_dir / "selected_manifest.json"
                ),
                "locked_report_accessed": False,
            },
            "trainable": {
                "trainable_parameters": parameter_count,
            },
            "capacity_role": "matched_target_support_audit_candidate",
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "final_writer_target_contract_sealed": False,
        }
        (output / "fit_selection_result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_output_checksums(output)
        return output

    def _outputs(self, root: Path):
        reductions = {
            "last_two_qv_r8": {3: 0.02, 4: 0.01},
            "all_expert_qv_r8": {3: 0.05, 4: 0.04},
            "official_default_r8": {3: 0.04, 4: 0.03},
        }
        return {
            (variant, task_id): self._fit_output(
                root,
                variant=variant,
                task_id=task_id,
                reduction=reductions[variant][task_id],
            )
            for variant in self.variants
            for task_id in (3, 4)
        }

    def test_grant_freezes_six_states_before_closed_loop_or_report_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = self._outputs(root)
            grant_path = root / "screening_freeze" / "screening_grant.json"

            grant = create_support_screening_grant(
                config_path=self.config,
                parent_path=self.parent,
                phase0_path=self.phase0,
                competence_path=self.competence,
                fit_outputs=outputs,
                grant_path=grant_path,
            )

            self.assertEqual(
                grant["status"],
                "target_support_fit_selections_frozen_before_closed_loop_screening",
            )
            self.assertTrue(grant["screening_rollout_authorized"])
            self.assertFalse(grant["locked_report_access_authorized"])
            self.assertFalse(grant["rank16_authorized"])
            self.assertFalse(grant["writer_authorized"])
            self.assertNotIn("screening_stage", grant)
            self.assertEqual(grant["query_ranking"][0]["variant"], "all_expert_qv_r8")
            self.assertEqual(grant["screening_init_state_indices"], list(range(24, 32)))
            self.assertEqual(len(grant["fit_evidence"]), 6)
            self.assertEqual(
                validate_support_screening_grant(
                    grant_path=grant_path,
                    config_path=self.config,
                    parent_path=self.parent,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    fit_outputs=outputs,
                ),
                grant,
            )

    def test_missing_or_changed_fit_fails_before_publishing_grant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = self._outputs(root)
            outputs.pop(("official_default_r8", 4))
            grant_path = root / "missing" / "screening_grant.json"
            with self.assertRaisesRegex(Exception, "exact frozen fit-output set"):
                create_support_screening_grant(
                    config_path=self.config,
                    parent_path=self.parent,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    fit_outputs=outputs,
                    grant_path=grant_path,
                )
            self.assertFalse(grant_path.parent.exists())

    def test_rank16_grant_freezes_only_two_states_and_fresh_screening_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ROOT / "configs" / "gate_zero_target_support_rank16.toml"
            variant = "official_default_r16"
            outputs = {
                (variant, task_id): self._fit_output(
                    root,
                    variant=variant,
                    task_id=task_id,
                    reduction=0.05,
                    config_path=config,
                    parameter_count=742656,
                )
                for task_id in (3, 4)
            }
            grant_path = root / "rank16_freeze" / "screening_grant.json"

            grant = create_support_screening_grant(
                config_path=config,
                parent_path=self.parent,
                phase0_path=self.phase0,
                competence_path=self.competence,
                fit_outputs=outputs,
                grant_path=grant_path,
            )

            self.assertEqual(grant["variants"], [variant])
            self.assertEqual(grant["screening_stage"], "rank16")
            self.assertEqual(grant["screening_init_state_indices"], list(range(32, 40)))
            self.assertEqual(len(grant["fit_evidence"]), 2)
            self.assertFalse(grant["rank16_authorized"])
            self.assertEqual(
                validate_support_screening_grant(
                    grant_path=grant_path,
                    config_path=config,
                    parent_path=self.parent,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                    fit_outputs=outputs,
                ),
                grant,
            )

    def _screening_grant(self, *, reduction: float) -> dict[str, object]:
        evidence = {}
        ranking = []
        for order, variant in enumerate(self.variants):
            ranking.append(
                {
                    "variant": variant,
                    "positive_query_task_count": 2,
                    "median_query_reduction_fraction": reduction + order * 0.01,
                    "trainable_parameters": self.parameter_counts[variant],
                }
            )
            for task_id in (3, 4):
                evidence[f"{variant}:task{task_id}"] = {
                    "selected_trainable_state_sha256": str(order + 1) * 64,
                    "selected_query_metrics": {
                        "query_flow_mse": 1.0 - reduction - order * 0.01,
                        "base_query_flow_mse": 1.0,
                        "action_drift_proxy": 0.01,
                    },
                }
        return {
            "fit_evidence": evidence,
            "query_ranking": list(reversed(ranking)),
        }

    def _screening_arms(self, successes: dict[str, int]) -> list[dict[str, object]]:
        arms = []
        for task_id in (3, 4):
            for condition in ["frozen_base", *self.variants]:
                count = 2 if condition == "frozen_base" else successes[condition]
                arms.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "mechanics_valid": True,
                        "successes": [True] * count + [False] * (8 - count),
                        "official_rollout_init_state_indices": list(range(24, 32)),
                        "seeds": list(range(5500, 5508)),
                    }
                )
        return arms

    def test_screening_shards_cover_base_and_three_supports_without_duplication(self) -> None:
        shards = canonical_support_screening_shards()

        self.assertEqual(
            shards,
            [
                [(3, "frozen_base"), (3, "last_two_qv_r8")],
                [(3, "all_expert_qv_r8"), (3, "official_default_r8")],
                [(4, "frozen_base"), (4, "last_two_qv_r8")],
                [(4, "all_expert_qv_r8"), (4, "official_default_r8")],
            ],
        )
        assigned = [
            assigned_support_screening_arms(rank=rank, world_size=4)
            for rank in range(4)
        ]
        self.assertEqual(assigned, shards)

    def test_rank16_screen_uses_two_task_shards_without_duplication(self) -> None:
        variants = ["official_default_r16"]
        shards = canonical_support_screening_shards(
            variants=variants, task_ids=[3, 4]
        )

        self.assertEqual(
            shards,
            [
                [(3, "frozen_base"), (3, "official_default_r16")],
                [(4, "frozen_base"), (4, "official_default_r16")],
            ],
        )
        self.assertEqual(
            [
                assigned_support_screening_arms(
                    rank=rank,
                    world_size=2,
                    variants=variants,
                    task_ids=[3, 4],
                )
                for rank in range(2)
            ],
            shards,
        )

    def test_smallest_rank8_support_passing_all_frozen_thresholds_is_selected(self) -> None:
        decision = decide_support_screening(
            arms=self._screening_arms(
                {
                    "last_two_qv_r8": 4,
                    "all_expert_qv_r8": 5,
                    "official_default_r8": 5,
                }
            ),
            grant=self._screening_grant(reduction=0.22),
            variants=self.variants,
            task_ids=[3, 4],
            parameter_counts=self.parameter_counts,
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 0.70,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
            },
            expected_init_state_indices=list(range(24, 32)),
            expected_seeds=list(range(5500, 5508)),
        )

        self.assertEqual(decision["status"], "rank8_support_selected_pending_confirmation")
        self.assertEqual(decision["selected_variant"], "last_two_qv_r8")
        self.assertTrue(decision["confirmation_authorized"])
        self.assertFalse(decision["rank16_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_no_rank8_pass_authorizes_only_one_rank16_scope(self) -> None:
        decision = decide_support_screening(
            arms=self._screening_arms(
                {
                    "last_two_qv_r8": 2,
                    "all_expert_qv_r8": 3,
                    "official_default_r8": 2,
                }
            ),
            grant=self._screening_grant(reduction=0.01),
            variants=self.variants,
            task_ids=[3, 4],
            parameter_counts=self.parameter_counts,
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 0.70,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
            },
            expected_init_state_indices=list(range(24, 32)),
            expected_seeds=list(range(5500, 5508)),
        )

        self.assertEqual(decision["status"], "rank8_support_screen_failed_rank16_authorized")
        self.assertIsNone(decision["selected_variant"])
        self.assertFalse(decision["confirmation_authorized"])
        self.assertTrue(decision["rank16_authorized"])
        self.assertEqual(decision["rank16_scope"], "official_default_r8")
        self.assertFalse(decision["writer_authorized"])

    def test_failed_rank16_screen_cannot_authorize_another_rank_search(self) -> None:
        variant = "official_default_r16"
        grant = {
            "fit_evidence": {
                f"{variant}:task{task_id}": {
                    "selected_query_metrics": {
                        "query_flow_mse": 0.96,
                        "base_query_flow_mse": 1.0,
                        "action_drift_proxy": 0.01,
                    }
                }
                for task_id in (3, 4)
            },
            "query_ranking": [{"variant": variant}],
        }
        arms = []
        for task_id in (3, 4):
            for condition, count in (("frozen_base", 2), (variant, 3)):
                arms.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "mechanics_valid": True,
                        "successes": [True] * count + [False] * (8 - count),
                        "official_rollout_init_state_indices": list(range(32, 40)),
                        "seeds": list(range(5600, 5608)),
                    }
                )

        decision = decide_support_screening(
            arms=arms,
            grant=grant,
            variants=[variant],
            task_ids=[3, 4],
            parameter_counts={variant: 742656},
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 0.70,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
            },
            expected_init_state_indices=list(range(32, 40)),
            expected_seeds=list(range(5600, 5608)),
            rank_stage="rank16",
        )

        self.assertEqual(decision["status"], "rank16_support_screen_failed")
        self.assertIsNone(decision["selected_variant"])
        self.assertFalse(decision["confirmation_authorized"])
        self.assertFalse(decision["rank16_authorized"])
        self.assertIsNone(decision["rank16_scope"])
        self.assertFalse(decision["writer_authorized"])

    def test_support_state_authority_never_swaps_or_opens_report_state(self) -> None:
        self.assertEqual(support_state_authority(3, "frozen_base"), (None, None))
        for variant in self.variants:
            self.assertEqual(support_state_authority(3, variant), (variant, 3))
        with self.assertRaisesRegex(Exception, "condition"):
            support_state_authority(3, "swapped_adapter")
        with self.assertRaisesRegex(Exception, "task"):
            support_state_authority(90, self.variants[0])

    def test_screening_launcher_has_one_freeze_and_four_rank_rollout_path(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_target_support_screen.sh"),
                "--gpus=4,5,6,7",
                "--fit-root=/tmp/ember-support-fit-root",
                "--screening-freeze-dir=/tmp/ember-support-freeze",
                "--output-dir=/tmp/ember-support-screen",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_support.screen", completed.stdout)
        self.assertIn("-m ember.gate_zero_support.screen_runtime", completed.stdout)
        self.assertIn("--nproc-per-node=4", completed.stdout)
        self.assertNotIn("gate_zero_oracle_report_runtime", completed.stdout)

    def test_screening_launcher_accepts_rank16_config_and_two_rank_topology(self) -> None:
        config = ROOT / "configs" / "gate_zero_target_support_rank16.toml"
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_target_support_screen.sh"),
                f"--config={config}",
                "--gpus=4,5",
                "--fit-root=/tmp/ember-support-rank16-fit-root",
                "--screening-freeze-dir=/tmp/ember-support-rank16-freeze",
                "--output-dir=/tmp/ember-support-rank16-screen",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(config), completed.stdout)
        self.assertIn("--nproc-per-node=2", completed.stdout)


if __name__ == "__main__":
    unittest.main()
