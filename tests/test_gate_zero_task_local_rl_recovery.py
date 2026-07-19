from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_task_local_rl.contract import (  # noqa: E402
    GateZeroTaskLocalRLContractError,
    assigned_task_local_rl_arm,
    decide_task_local_rl_node,
    load_task_local_rl_spec,
    normalized_episode_advantages,
    validate_task_local_rl_spec,
)
from ember.gate_zero_task_local_rl.runtime import (  # noqa: E402
    AnchorRecordingEnvPreprocessor,
    ExplorationActionProcessor,
    balanced_anchor_slots,
    build_balanced_replay_batch,
    signed_flow_ratio_loss,
    validated_flow_action_shape,
    validate_training_reset_events,
)


class GateZeroTaskLocalRLRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_task_local_rl_recovery.toml"
        cls.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        cls.phase0 = ROOT / "configs" / "phase0.toml"
        cls.fit = ROOT / "configs" / "gate_zero_mature_lora_lr_recovery.toml"
        cls.headroom = ROOT / "configs" / "gate_zero_mature_lora_headroom_screen.toml"
        cls.diagnostic = ROOT / "configs" / "gate_zero_mature_lora_candidate_step_diagnostic.toml"
        cls.launcher = ROOT / "scripts" / "run_gate_zero_task_local_rl_recovery.sh"
        cls.spec = load_task_local_rl_spec(
            cls.path,
            gate_zero_path=cls.gate_zero,
            phase0_path=cls.phase0,
            fit_path=cls.fit,
            headroom_path=cls.headroom,
            diagnostic_path=cls.diagnostic,
        )

    def test_contract_is_same_task_four_arm_lora_only_recovery(self) -> None:
        self.assertEqual(self.spec["task_ids"], [3, 4])
        self.assertEqual(self.spec["initializations"], ["zero_init", "supervised_init"])
        self.assertEqual(self.spec["lora"]["target_count"], 37)
        self.assertEqual(self.spec["lora"]["rank"], 32)
        self.assertEqual(self.spec["lora"]["trainable_parameters"], 1_485_312)
        self.assertFalse(self.spec["authority"]["writer_present"])
        self.assertFalse(self.spec["algorithm"]["shared_parameter_updates"])
        self.assertFalse(self.spec["algorithm"]["writer_updates"])
        self.assertEqual(self.spec["training_interaction"]["interaction_episode_nodes"], [16])
        self.assertEqual(self.spec["exploration"]["standard_deviation"], [0.05] * 6 + [0.0])
        self.assertEqual(
            self.spec["algorithm"]["name"],
            "per_sample_conditional_flow_loss_ratio_with_signed_episode_advantage",
        )
        self.assertTrue(self.spec["algorithm"]["not_full_fpo_plus"])

    def test_four_rank_assignment_has_no_duplicate_or_idle_arm(self) -> None:
        assignments = [assigned_task_local_rl_arm(rank=i, world_size=4, spec=self.spec) for i in range(4)]
        self.assertEqual(
            assignments,
            [(3, "zero_init"), (3, "supervised_init"), (4, "zero_init"), (4, "supervised_init")],
        )
        self.assertEqual(len(set(assignments)), 4)

    def test_episode_advantages_are_signed_normalized_and_fail_on_flat_reward(self) -> None:
        advantages = normalized_episode_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(advantages.std(unbiased=False)), 1.0, places=6)
        self.assertGreater(float(advantages[0]), 0.0)
        self.assertLess(float(advantages[1]), 0.0)
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            normalized_episode_advantages(torch.zeros(8))

    def _metrics(self, *, zero=(2, 1), supervised=(2, 1), drift=0.01) -> dict:
        return {
            "mechanics_valid": True,
            "maximum_saturation_fraction": 0.01,
            "nonfinite_count": 0,
            "action_drift_by_arm": {"zero_init_rl": drift, "supervised_init_rl": drift},
            "paired_net_wins_by_arm": {
                "zero_init_rl": list(zero),
                "supervised_init_rl": list(supervised),
            },
        }

    def test_stage16_passes_early_or_stops_without_budget_extension(self) -> None:
        passed = decide_task_local_rl_node(self.spec, interaction_episodes=16, metrics=self._metrics())
        self.assertEqual(passed["status"], "rl_candidate_selected_for_fresh_gate")
        self.assertEqual(passed["selected_interaction_episodes"], 16)
        stopped = decide_task_local_rl_node(
            self.spec, interaction_episodes=16, metrics=self._metrics(zero=(0, 0), supervised=(0, 0))
        )
        self.assertEqual(stopped["status"], "task_local_rl_early_check_not_supported")

    def test_stage32_is_not_an_active_decision_node(self) -> None:
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            decide_task_local_rl_node(
                self.spec, interaction_episodes=32, metrics=self._metrics(zero=(1, 0))
            )

    def test_threshold_or_task_mutation_fails_closed(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["candidate_decision"]["minimum_median_success_gain_pp"] = 12.5
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )
        changed = copy.deepcopy(self.spec)
        changed["mechanical_recovery"]["optimizer_updates_before_failure"] = 1
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )
        changed = copy.deepcopy(self.spec)
        changed["flow_shape_recovery"]["model_action_shape_after_preprocessing"] = [64, 50, 7]
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )
        changed = copy.deepcopy(self.spec)
        changed["signed_flow_ratio_recovery"]["parent_awr_result_sha256"] = "0" * 64
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )

    def test_balanced_anchor_slots_are_fixed_count_and_cover_endpoints(self) -> None:
        self.assertEqual(balanced_anchor_slots(8, slots=8), list(range(8)))
        short = balanced_anchor_slots(3, slots=8)
        self.assertEqual(len(short), 8)
        self.assertEqual(short[0], 0)
        self.assertEqual(short[-1], 2)
        self.assertTrue(all(0 <= value < 3 for value in short))

    def test_exploration_processor_is_common_random_and_clips(self) -> None:
        def identity(value):
            return value

        kwargs = {
            "base": identity,
            "standard_deviation": [0.05] * 6 + [0.0],
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "seed": 123,
        }
        left = ExplorationActionProcessor(**kwargs)
        right = ExplorationActionProcessor(**kwargs)
        action = torch.tensor([[0.99] * 7, [-0.99] * 7])
        first = left({"action": action.clone()})["action"]
        second = right({"action": action.clone()})["action"]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first <= 1.0))
        self.assertTrue(torch.all(first >= -1.0))
        self.assertGreater(left.total_scalars, 0)
        self.assertGreaterEqual(left.saturation_fraction, 0.0)
        self.assertEqual(sum(left.saturated_scalars_by_dimension), left.saturated_scalars)
        self.assertEqual(len(left.saturated_scalars_by_dimension), 7)
        self.assertEqual(left.total_scalars, 12)
        self.assertEqual(left.saturated_scalars_by_dimension[-1], 0)
        self.assertTrue(torch.equal(first[:, -1], action[:, -1]))

    def test_replay_builder_balances_episodes_and_pads_chunks(self) -> None:
        batch_size = 2
        anchors = []
        for step in (0, 50):
            anchors.append(
                {
                    "step": step,
                    "observation.images.camera1": torch.full(
                        (batch_size, 3, 4, 4), step // 50, dtype=torch.uint8
                    ),
                    "observation.images.camera2": torch.zeros(
                        batch_size, 3, 4, 4, dtype=torch.uint8
                    ),
                    "observation.state": torch.zeros(batch_size, 8),
                    "task": ["task"] * batch_size,
                }
            )
        actions = torch.arange(batch_size * 70 * 7, dtype=torch.float32).reshape(
            batch_size, 70, 7
        )
        done = torch.zeros(batch_size, 70, dtype=torch.bool)
        done[0, 20:] = True
        done[1, 69:] = True
        success = torch.zeros(batch_size, 70, dtype=torch.bool)
        success[0, 20] = True
        rollout = {"action": actions, "done": done, "success": success}
        replay = build_balanced_replay_batch(
            anchors=anchors,
            rollout=rollout,
            seeds=[10, 11],
            task_id=3,
            action_chunk_size=50,
            anchors_per_episode=8,
        )
        self.assertEqual(replay["action"].shape, (16, 50, 7))
        self.assertEqual(replay["action_is_pad"].shape, (16, 50))
        self.assertEqual(replay["episode_return"].tolist(), [1.0] * 8 + [0.0] * 8)
        self.assertEqual(len(replay["row_keys"]), 16)
        self.assertEqual(len(set(replay["row_keys"])), 16)
        self.assertTrue(replay["action_is_pad"][:8, 21:].all())

    def test_anchor_recorder_keeps_only_replan_observations_as_uint8(self) -> None:
        recorder = AnchorRecordingEnvPreprocessor(base=lambda value: value, interval=50)
        batch = {
            "observation.images.camera1": torch.full((2, 3, 8, 8), 0.5),
            "observation.images.camera2": torch.ones(2, 3, 8, 8),
            "observation.state": torch.zeros(2, 8),
            "task": ["a", "a"],
        }
        for _ in range(51):
            recorder(copy.deepcopy(batch))
        self.assertEqual([value["step"] for value in recorder.anchors], [0, 50])
        self.assertEqual(recorder.anchors[0]["observation.images.camera1"].dtype, torch.uint8)
        self.assertEqual(
            int(recorder.anchors[0]["observation.images.camera1"][0, 0, 0, 0]), 128
        )

    def test_training_reset_identity_is_round_specific(self) -> None:
        events = [
            {"before": list(range(0, 8)), "after": list(range(8, 16)), "seeds": list(range(6200, 6208))},
            {"before": list(range(8, 16)), "after": list(range(16, 24)), "seeds": list(range(6208, 6216))},
        ]
        self.assertTrue(
            validate_training_reset_events(
                events,
                round_index=1,
                batch_size=8,
                seed_start=6200,
            )
        )
        changed = copy.deepcopy(events)
        changed[-1]["after"] = list(range(24, 32))
        self.assertFalse(
            validate_training_reset_events(
                changed,
                round_index=1,
                batch_size=8,
                seed_start=6200,
            )
        )

    def test_launcher_is_one_canonical_four_gpu_staged_path(self) -> None:
        text = self.launcher.read_text(encoding="utf-8")
        self.assertIn("--nproc-per-node=4", text)
        self.assertIn('[[ "$stop_after_episodes" == 16 ]]', text)
        self.assertIn("--resume", text)
        self.assertIn("gpu_telemetry_", text)
        self.assertNotIn("ppo", text.lower())
        self.assertNotIn("writer", text.lower())

    def test_signed_flow_ratio_loss_pushes_success_toward_and_failure_away(self) -> None:
        current = torch.tensor([1.0, 1.0], requires_grad=True)
        old = torch.tensor([1.0, 1.0])
        advantages = torch.tensor([1.0, -1.0])
        loss, metrics = signed_flow_ratio_loss(
            current,
            old,
            advantages,
            ratio_clip=0.02,
            negative_spo_penalty=0.01,
            log_ratio_clamp=5.0,
        )
        loss.backward()
        self.assertGreater(float(current.grad[0]), 0.0)
        self.assertLess(float(current.grad[1]), 0.0)
        self.assertAlmostEqual(metrics["ratio_mean"], 1.0, places=6)
        self.assertAlmostEqual(metrics["advantage_mean"], 0.0, places=6)
        with self.assertRaises(Exception):
            signed_flow_ratio_loss(
                torch.ones(2),
                torch.ones(3),
                torch.ones(2),
                ratio_clip=0.02,
                negative_spo_penalty=0.01,
                log_ratio_clamp=5.0,
            )

    def test_flow_noise_uses_processed_model_action_width(self) -> None:
        batch = {"action": torch.zeros(64, 50, 7)}
        self.assertEqual(
            validated_flow_action_shape(
                batch,
                expected_batch_size=64,
                expected_chunk_size=50,
                input_action_dim=7,
                model_action_dim=32,
            ),
            (50, 32),
        )
        with self.assertRaises(Exception):
            validated_flow_action_shape(
                {"action": torch.zeros(64, 50, 32)},
                expected_batch_size=64,
                expected_chunk_size=50,
                input_action_dim=7,
                model_action_dim=32,
            )


if __name__ == "__main__":
    unittest.main()
