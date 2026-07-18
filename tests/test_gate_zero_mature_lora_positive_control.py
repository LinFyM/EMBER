from __future__ import annotations

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
    load_recovery_artifact,
    save_recovery_artifact,
)
from ember.gate_zero_oracle_fit import (  # noqa: E402
    GateZeroOracleFitError,
    resolve_training_target_step,
    select_fixed_final_candidate,
)
from ember.gate_zero_oracle_session import (  # noqa: E402
    augment_support_images,
    build_oracle_optimizer,
    build_oracle_scheduler,
)
from ember.gate_zero_support.mature_contract import (  # noqa: E402
    GateZeroMatureControlContractError,
    load_mature_lora_positive_control_spec,
)
from ember.gate_zero_support.screen import decide_support_screening  # noqa: E402


class GateZeroMatureLoraPositiveControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_mature_lora_positive_control.toml"
        self.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"

    def load(self):
        return load_mature_lora_positive_control_spec(
            self.config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

    def test_checked_in_contract_is_mature_recipe_not_small_acquisition_claim(self) -> None:
        spec = self.load()
        variant = spec["fit"]["mature_official_default_r32"]

        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["fit"]["support_episode_bounds"], [0, 39])
        self.assertEqual(
            spec["fit"]["support_episode_roles"],
            ["writer_spec", "source_base_fit", "oracle_support"],
        )
        self.assertEqual(spec["selection"]["query_episode_bounds"], [40, 45])
        self.assertEqual(spec["fit"]["optimizer_steps"], 20_000)
        self.assertEqual(spec["fit"]["candidate_steps"][-1], 20_000)
        self.assertEqual(spec["selection"]["candidate_rule"], "fixed_final_optimizer_step")
        self.assertEqual(variant["rank"], 32)
        self.assertEqual(variant["alpha"], 16)
        self.assertEqual(variant["init_lora_weights"], "gaussian")
        self.assertEqual(variant["expected_trainable_parameters"], 1_485_312)
        self.assertEqual(len(variant["target_modules"]), 37)
        self.assertEqual(variant["scheduler"], "linear_warmup_cosine_decay")
        self.assertEqual(variant["warmup_steps"], 1_000)
        self.assertEqual(variant["decay_steps"], 20_000)
        self.assertEqual(variant["augmentation"], "random_resized_crop")
        self.assertEqual(spec["screening_rollout"]["init_state_indices"], list(range(40, 48)))
        self.assertTrue(spec["owner_override"]["small_recipe_final_negative_superseded"])
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])
        self.assertFalse(spec["writer_authorized_before_closed_loop"])

    def test_primary_source_or_prior_result_hash_drift_fails_closed(self) -> None:
        changed = self.config.read_text(encoding="utf-8").replace(
            "65b2abffcf8b2c7e8907c03f4e21cd8435da38b94afb9e8b41337a54bd323b00",
            "0" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mature.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(
                GateZeroMatureControlContractError, "rank-16 screening result"
            ):
                load_mature_lora_positive_control_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_fixed_final_selection_cannot_promote_an_earlier_query_winner(self) -> None:
        candidates = [
            {"step": 0, "query_flow_mse": 1.0, "action_drift_proxy": 0.0},
            {"step": 10_000, "query_flow_mse": 0.2, "action_drift_proxy": 0.4},
            {"step": 20_000, "query_flow_mse": 0.3, "action_drift_proxy": 0.5},
        ]

        selected = select_fixed_final_candidate(candidates, final_step=20_000)

        self.assertEqual(selected["step"], 20_000)

    def test_staged_stop_must_be_a_future_predeclared_candidate(self) -> None:
        candidates = list(range(0, 20_001, 1_000))
        self.assertEqual(
            resolve_training_target_step(
                start_step=0,
                optimizer_steps=20_000,
                candidate_steps=candidates,
                stop_after_step=1_000,
            ),
            1_000,
        )
        self.assertEqual(
            resolve_training_target_step(
                start_step=1_000,
                optimizer_steps=20_000,
                candidate_steps=candidates,
                stop_after_step=2_000,
            ),
            2_000,
        )
        with self.assertRaises(GateZeroOracleFitError):
            resolve_training_target_step(
                start_step=0,
                optimizer_steps=20_000,
                candidate_steps=candidates,
                stop_after_step=500,
            )
        with self.assertRaises(GateZeroOracleFitError):
            resolve_training_target_step(
                start_step=1_000,
                optimizer_steps=20_000,
                candidate_steps=candidates,
                stop_after_step=1_000,
            )

    def test_training_crop_is_deterministic_shape_preserving_and_step_conditioned(self) -> None:
        base = torch.arange(2 * 3 * 16 * 16, dtype=torch.uint8).reshape(2, 3, 16, 16)
        batch = {
            "observation.images.camera1": base.clone(),
            "observation.images.camera2": base.flip(-1).clone(),
        }
        row_keys = ["task3/demo0/frame0", "task3/demo0/frame1"]

        first = augment_support_images(
            {key: value.clone() for key, value in batch.items()},
            row_keys=row_keys,
            optimizer_step=7,
            seed=123,
            scale_min=0.9,
            scale_max=1.0,
        )
        repeated = augment_support_images(
            {key: value.clone() for key, value in batch.items()},
            row_keys=row_keys,
            optimizer_step=7,
            seed=123,
            scale_min=0.9,
            scale_max=1.0,
        )
        changed = augment_support_images(
            {key: value.clone() for key, value in batch.items()},
            row_keys=row_keys,
            optimizer_step=8,
            seed=123,
            scale_min=0.9,
            scale_max=1.0,
        )

        for key in batch:
            self.assertEqual(first[key].shape, batch[key].shape)
            self.assertEqual(first[key].dtype, torch.float32)
            self.assertTrue(torch.equal(first[key], repeated[key]))
        self.assertTrue(any(not torch.equal(first[key], changed[key]) for key in batch))

    def test_scheduler_state_is_part_of_atomic_long_run_recovery(self) -> None:
        spec = self.load()
        variant = spec["fit"]["mature_official_default_r32"]
        model = torch.nn.Linear(3, 2)
        optimizer = build_oracle_optimizer(model, variant)
        scheduler = build_oracle_scheduler(optimizer, variant, optimizer_steps=20_000)
        self.assertIsNotNone(scheduler)
        for _ in range(3):
            model(torch.ones(1, 3)).square().sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        expected_scheduler = scheduler.state_dict()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery = save_recovery_artifact(
                root,
                variant="mature_official_default_r32",
                task_id=3,
                step=3,
                trainable_state={
                    name: value.detach().clone() for name, value in model.named_parameters()
                },
                optimizer=optimizer,
                scheduler=scheduler,
                authorities={"execution_contract_sha256": "c" * 64},
            )
            fresh_model = torch.nn.Linear(3, 2)
            fresh_optimizer = build_oracle_optimizer(fresh_model, variant)
            fresh_scheduler = build_oracle_scheduler(
                fresh_optimizer, variant, optimizer_steps=20_000
            )

            restored = load_recovery_artifact(
                recovery,
                model=fresh_model,
                optimizer=fresh_optimizer,
                scheduler=fresh_scheduler,
            )

            self.assertEqual(restored, 3)
            self.assertEqual(fresh_scheduler.state_dict(), expected_scheduler)

    @staticmethod
    def _arm(task_id: int, condition: str, successes: int) -> dict[str, object]:
        return {
            "task_id": task_id,
            "condition": condition,
            "successes": [True] * successes + [False] * (8 - successes),
            "mechanics_valid": True,
            "official_rollout_init_state_indices": list(range(40, 48)),
            "seeds": list(range(5800, 5808)),
        }

    def test_mature_positive_control_pass_seals_gate0_without_a_second_confirmation(self) -> None:
        variant = "mature_official_default_r32"
        grant = {
            "fit_evidence": {
                f"{variant}:task3": {
                    "selected_query_metrics": {
                        "base_query_flow_mse": 1.0,
                        "query_flow_mse": 0.70,
                        "action_drift_proxy": 0.4,
                    }
                },
                f"{variant}:task4": {
                    "selected_query_metrics": {
                        "base_query_flow_mse": 1.0,
                        "query_flow_mse": 0.60,
                        "action_drift_proxy": 0.5,
                    }
                },
            }
        }
        arms = [
            self._arm(3, "frozen_base", 2),
            self._arm(3, variant, 5),
            self._arm(4, "frozen_base", 1),
            self._arm(4, variant, 4),
        ]

        decision = decide_support_screening(
            arms=arms,
            grant=grant,
            variants=[variant],
            task_ids=[3, 4],
            parameter_counts={variant: 1_485_312},
            thresholds={
                "median_success_gain_pp_min": 15.0,
                "median_locked_action_loss_reduction_fraction_min": 0.20,
                "positive_task_fraction_min": 1.0,
                "median_selection_drift_proxy_max": 0.02,
                "two_task_positive_count_required": 2,
                "selection_drift_is_diagnostic_only": True,
            },
            expected_init_state_indices=list(range(40, 48)),
            expected_seeds=list(range(5800, 5808)),
            rank_stage="mature_positive_control",
        )

        self.assertEqual(decision["status"], "mature_lora_positive_control_passed")
        self.assertEqual(decision["selected_variant"], variant)
        self.assertFalse(decision["confirmation_authorized"])
        self.assertTrue(decision["gate_zero_authorized"])
        self.assertTrue(decision["writer_authorized"])
        self.assertTrue(decision["final_writer_target_contract_sealed"])

    def test_existing_launchers_accept_mature_contract_without_a_parallel_path(self) -> None:
        fit = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_fit.sh"),
                f"--config={self.config}",
                "--variant=mature_official_default_r32",
                "--task-id=3",
                "--gpu=4",
                "--output-dir=/tmp/ember-mature-fit",
                "--latest-link=/tmp/ember-mature-fit-latest",
                "--stop-after-step=1000",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )
        screen = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_target_support_screen.sh"),
                f"--config={self.config}",
                "--gpus=4,5",
                "--fit-root=/tmp/ember-mature-fit-root",
                "--screening-freeze-dir=/tmp/ember-mature-freeze",
                "--output-dir=/tmp/ember-mature-screen",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
        )

        self.assertEqual(fit.returncode, 0, fit.stderr)
        self.assertIn("-m ember.gate_zero_oracle_fit", fit.stdout)
        self.assertIn("--stop-after-step 1000", fit.stdout)
        self.assertEqual(screen.returncode, 0, screen.stderr)
        self.assertIn("-m ember.gate_zero_support.screen_runtime", screen.stdout)


if __name__ == "__main__":
    unittest.main()
