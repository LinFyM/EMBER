from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_oracle_artifacts import (  # noqa: E402
    load_recovery_artifact,
    save_recovery_artifact,
)
from ember.gate_zero_oracle_contract import load_oracle_fit_spec  # noqa: E402
from ember.gate_zero_oracle_fit import (  # noqa: E402
    GateZeroOracleFitError,
    resolve_training_target_step,
    select_fixed_final_candidate,
)
from ember.gate_zero_oracle_session import (  # noqa: E402
    MATURE_SUPPORT_QUERY_STAGES,
    augment_support_images,
    build_oracle_optimizer,
    build_oracle_scheduler,
)
from ember.gate_zero_support.mature_contract import (  # noqa: E402
    GateZeroMatureControlContractError,
    load_mature_lora_positive_control_spec,
)
from ember.gate_zero_support.mature_lora_lr_contract import (  # noqa: E402
    GateZeroMatureLoraLRContractError,
)
from ember.gate_zero_support.screen import decide_support_screening  # noqa: E402


class GateZeroMatureLoraPositiveControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_mature_lora_positive_control.toml"
        self.recovery_config = (
            ROOT / "configs" / "gate_zero_mature_lora_all_linear_recovery.toml"
        )
        self.upper_bound_config = (
            ROOT / "configs" / "gate_zero_mature_action_expert_upper_bound.toml"
        )
        self.upper_bound_ladder = (
            ROOT / "configs" / "gate_zero_mature_action_expert_stage_ladder.toml"
        )
        self.action_expert_lr_recovery_config = (
            ROOT / "configs" / "gate_zero_mature_action_expert_lr_recovery.toml"
        )
        self.action_expert_lr_recovery_ladder = (
            ROOT / "configs" / "gate_zero_mature_action_expert_lr_recovery_ladder.toml"
        )
        self.lora_lr_recovery_config = (
            ROOT / "configs" / "gate_zero_mature_lora_lr_recovery.toml"
        )
        self.lora_lr_recovery_ladder = (
            ROOT / "configs" / "gate_zero_mature_lora_lr_recovery_ladder.toml"
        )
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

    def load_recovery(self):
        return load_mature_lora_positive_control_spec(
            self.recovery_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

    def load_upper_bound(self):
        return load_mature_lora_positive_control_spec(
            self.upper_bound_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

    def load_action_expert_lr_recovery(self):
        return load_mature_lora_positive_control_spec(
            self.action_expert_lr_recovery_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

    def test_action_expert_lr_recovery_is_one_bounded_magnitude_test(self) -> None:
        spec = self.load_action_expert_lr_recovery()
        variant_name = "mature_action_expert_lr25e6_recovery"
        variant = spec["fit"][variant_name]

        self.assertEqual(spec["variants"], [variant_name])
        self.assertEqual(spec["screening_stage"], "mature_capacity_lr_recovery")
        self.assertIn(spec["screening_stage"], MATURE_SUPPORT_QUERY_STAGES)
        self.assertEqual(variant["adaptation_kind"], "partial_update")
        self.assertEqual(variant["expected_trainable_parameters"], 99_880_992)
        self.assertEqual(variant["learning_rate"], 0.000025)
        self.assertEqual(variant["warmup_steps"], 1_000)
        self.assertEqual(variant["decay_learning_rate"], 0.000000625)
        self.assertEqual(
            spec["fit"]["candidate_steps"],
            [0, 250, 500, 750, 1_000, 2_000, 5_000, 10_000, 20_000],
        )
        self.assertEqual(
            spec["authority"]["task3_update_scale_probe_result_sha256"],
            "763f2e79fa844054ca7317b1d1f277567778d86d07050074e3a2f29b38d78189",
        )
        self.assertEqual(
            spec["authority"]["task4_update_scale_probe_result_sha256"],
            "39618a1104d526c31f90813d85aed613e10237d3f88a1213de44482319aeb3e1",
        )
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])
        self.assertFalse(spec["writer_authorized_before_closed_loop"])

        with self.action_expert_lr_recovery_ladder.open("rb") as handle:
            ladder = tomllib.load(handle)
        self.assertEqual(
            ladder["parent_fit_contract_sha256"],
            hashlib.sha256(self.action_expert_lr_recovery_config.read_bytes()).hexdigest(),
        )
        self.assertEqual(ladder["stage_steps"], [250, 500, 750, 1_000])
        self.assertEqual(
            ladder["continuation"]["step750_to_1000"],
            {
                "minimum_median_query_reduction_fraction": 0.01,
                "minimum_each_task_query_reduction_fraction": 0.0,
            },
        )
        self.assertEqual(
            ladder["success_at_step1000"]["minimum_median_query_reduction_fraction"],
            0.02,
        )
        self.assertTrue(ladder["success_at_step1000"]["authorizes_matched_lora_schedule"])
        self.assertFalse(ladder["success_at_step1000"]["authorizes_gate_zero"])

    def test_action_expert_lr_recovery_drift_fails_closed(self) -> None:
        changed = self.action_expert_lr_recovery_config.read_text(encoding="utf-8").replace(
            "learning_rate = 0.000025", "learning_rate = 0.00005"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lr-recovery.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroMatureControlContractError):
                load_mature_lora_positive_control_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_canonical_oracle_fit_dispatch_accepts_action_expert_lr_recovery(self) -> None:
        spec = load_oracle_fit_spec(
            self.action_expert_lr_recovery_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

        self.assertEqual(spec["variants"], ["mature_action_expert_lr25e6_recovery"])

    def test_matched_lora_lr_recovery_reuses_the_authorized_schedule(self) -> None:
        spec = load_oracle_fit_spec(
            self.lora_lr_recovery_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )
        variant_name = "mature_official_default_r32_lr25e6_recovery"
        variant = spec["fit"][variant_name]

        self.assertEqual(spec["variants"], [variant_name])
        self.assertEqual(spec["screening_stage"], "mature_lora_lr_recovery")
        self.assertIn(spec["screening_stage"], MATURE_SUPPORT_QUERY_STAGES)
        self.assertEqual(variant["adaptation_kind"], "lora")
        self.assertEqual(variant["rank"], 32)
        self.assertEqual(variant["alpha"], 16)
        self.assertEqual(variant["expected_trainable_parameters"], 1_485_312)
        self.assertEqual(len(variant["target_modules"]), 37)
        self.assertEqual(variant["learning_rate"], 0.000025)
        self.assertEqual(variant["decay_learning_rate"], 0.000000625)
        self.assertEqual(
            spec["fit"]["candidate_steps"],
            [0, 250, 500, 750, 1_000, 2_000, 5_000, 10_000, 20_000],
        )
        self.assertFalse(spec["screening_rollout"]["access_authorized"])
        self.assertTrue(
            spec["screening_rollout"]["requires_headroom_safe_source_contract"]
        )
        self.assertFalse(spec["authority"]["validation_numeric_access"])
        self.assertFalse(spec["authority"]["held_numeric_access"])

        with self.lora_lr_recovery_ladder.open("rb") as handle:
            ladder = tomllib.load(handle)
        self.assertEqual(
            ladder["parent_fit_contract_sha256"],
            hashlib.sha256(self.lora_lr_recovery_config.read_bytes()).hexdigest(),
        )
        self.assertEqual(ladder["stage_steps"], [250, 500, 750, 1_000])
        self.assertFalse(ladder["success_at_step1000"]["authorizes_gate_zero"])
        self.assertTrue(
            ladder["success_at_step1000"]["authorizes_headroom_safe_source_rollout_contract"]
        )

    def test_matched_lora_lr_recovery_drift_fails_closed(self) -> None:
        changed = self.lora_lr_recovery_config.read_text(encoding="utf-8").replace(
            "learning_rate = 0.000025", "learning_rate = 0.00005"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lora-lr-recovery.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroMatureLoraLRContractError):
                load_oracle_fit_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_mature_action_expert_upper_bound_is_non_matched_and_cannot_authorize_writer(self) -> None:
        primary = self.load()
        upper = self.load_upper_bound()
        variant_name = "mature_action_expert_upper_bound"
        variant = upper["fit"][variant_name]

        self.assertEqual(upper["variants"], [variant_name])
        self.assertEqual(variant["adaptation_kind"], "partial_update")
        self.assertEqual(variant["expected_trainable_parameters"], 99_880_992)
        self.assertFalse(variant["matched_baseline"])
        self.assertFalse(variant["may_authorize_gate_zero"])
        self.assertFalse(variant["may_authorize_writer"])
        self.assertFalse(variant["may_seal_writer_target_contract"])
        primary_variant = primary["fit"]["mature_official_default_r32"]
        for key in (
            "optimizer",
            "learning_rate",
            "betas",
            "epsilon",
            "weight_decay",
            "gradient_clip_norm",
            "scheduler",
            "warmup_steps",
            "decay_steps",
            "decay_learning_rate",
            "augmentation",
            "augmentation_scale_min",
            "augmentation_scale_max",
            "augmentation_seed",
            "seed",
        ):
            self.assertEqual(variant[key], primary_variant[key])
        self.assertFalse(upper["prior_lora_failure"]["continuation_to_step5000"])
        self.assertFalse(upper["authority"]["validation_numeric_access"])
        self.assertFalse(upper["authority"]["held_numeric_access"])
        self.assertIn(upper["screening_stage"], MATURE_SUPPORT_QUERY_STAGES)

        with self.upper_bound_ladder.open("rb") as handle:
            ladder = tomllib.load(handle)
        self.assertEqual(
            ladder["parent_fit_contract_sha256"],
            hashlib.sha256(self.upper_bound_config.read_bytes()).hexdigest(),
        )
        self.assertTrue(ladder["non_matched_capacity_diagnostic_only"])
        self.assertFalse(ladder["may_authorize_gate_zero"])
        self.assertFalse(ladder["may_authorize_writer"])
        self.assertEqual(
            ladder["continuation"]["step2000_to_5000"],
            {
                "minimum_median_query_reduction_fraction": 0.05,
                "minimum_each_task_query_reduction_fraction": 0.0,
                "maximum_median_regression_from_prior_fraction": 0.01,
            },
        )

    def test_mature_action_expert_upper_bound_authority_drift_fails_closed(self) -> None:
        changed = self.upper_bound_config.read_text(encoding="utf-8").replace(
            "may_authorize_writer = false", "may_authorize_writer = true"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "upper.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroMatureControlContractError):
                load_mature_lora_positive_control_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_canonical_oracle_fit_dispatch_accepts_mature_action_expert_upper_bound(self) -> None:
        spec = load_oracle_fit_spec(
            self.upper_bound_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

        self.assertEqual(spec["variants"], ["mature_action_expert_upper_bound"])

    def test_conditional_recovery_is_exact_all_action_expert_linear_same_recipe(self) -> None:
        primary = self.load()
        recovery = self.load_recovery()
        variant_name = "all_action_expert_linear_r32_same_recipe"
        variant = recovery["fit"][variant_name]
        targets = set(variant["target_modules"])
        expert = {
            target
            for target in targets
            if target.startswith("model.vlm_with_expert.lm_expert.layers.")
        }

        self.assertEqual(recovery["variants"], [variant_name])
        self.assertEqual(len(targets), 117)
        self.assertEqual(len(expert), 112)
        self.assertEqual(variant["expected_trainable_parameters"], 7_027_200)
        for layer in range(16):
            prefix = f"model.vlm_with_expert.lm_expert.layers.{layer}."
            self.assertEqual(
                {target.removeprefix(prefix) for target in expert if target.startswith(prefix)},
                {
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                },
            )
        primary_variant = primary["fit"]["mature_official_default_r32"]
        for key in (
            "rank",
            "alpha",
            "dropout",
            "init_lora_weights",
            "optimizer",
            "learning_rate",
            "betas",
            "weight_decay",
            "warmup_steps",
            "decay_steps",
            "augmentation",
            "augmentation_scale_min",
            "augmentation_scale_max",
        ):
            self.assertEqual(variant[key], primary_variant[key])
        self.assertEqual(
            recovery["authority"]["primary_mature_contract_sha256"],
            "882db40dca9ced15cf2b567f9fa57bf2c36c66e64654eef55c067d6485b4b259",
        )
        self.assertFalse(recovery["primary_failure"]["continuation_to_step10000"])
        self.assertTrue(recovery["bounded_recovery"]["no_further_target_or_rank_variants"])
        self.assertFalse(recovery["authority"]["validation_numeric_access"])
        self.assertFalse(recovery["authority"]["held_numeric_access"])

    def test_conditional_recovery_target_or_parent_drift_fails_closed(self) -> None:
        changed = self.recovery_config.read_text(encoding="utf-8").replace(
            '  "model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj",\n', ""
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recovery.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroMatureControlContractError):
                load_mature_lora_positive_control_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_canonical_oracle_fit_dispatch_accepts_conditional_recovery(self) -> None:
        spec = load_oracle_fit_spec(
            self.recovery_config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

        self.assertEqual(
            spec["variants"], ["all_action_expert_linear_r32_same_recipe"]
        )

    def test_conditional_recovery_ladder_keeps_primary_continuation_thresholds(self) -> None:
        with (
            ROOT / "configs" / "gate_zero_mature_lora_all_linear_stage_ladder.toml"
        ).open("rb") as handle:
            ladder = tomllib.load(handle)

        self.assertEqual(
            ladder["parent_fit_contract_sha256"],
            "82f5203ed86a25dac386bde68cb8a76efaba03c0f230fe2bd0249bb8d64fe15c",
        )
        self.assertTrue(ladder["only_target_support_differs_from_primary"])
        self.assertEqual(ladder["stage_steps"], [1_000, 2_000, 5_000, 10_000, 20_000])
        self.assertEqual(
            ladder["continuation"]["step5000_to_10000"],
            {
                "minimum_median_query_reduction_fraction": 0.10,
                "minimum_each_task_query_reduction_fraction": 0.02,
                "maximum_median_regression_from_prior_fraction": 0.01,
            },
        )
        self.assertIn(
            "another target/rank search", ladder["stop_and_diagnose"]["forbidden"]
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
