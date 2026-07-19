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
    episodic_awr_weights,
    load_task_local_rl_spec,
    validate_task_local_rl_spec,
)
from ember.gate_zero_task_local_rl.runtime import (  # noqa: E402
    AnchorRecordingEnvPreprocessor,
    ExplorationActionProcessor,
    balanced_anchor_slots,
    build_balanced_replay_batch,
    validate_training_reset_events,
    weighted_flow_loss,
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
        self.assertEqual(self.spec["training_interaction"]["interaction_episode_nodes"], [16, 32])
        self.assertEqual(self.spec["exploration"]["standard_deviation"], [0.05] * 6 + [0.0])

    def test_four_rank_assignment_has_no_duplicate_or_idle_arm(self) -> None:
        assignments = [assigned_task_local_rl_arm(rank=i, world_size=4, spec=self.spec) for i in range(4)]
        self.assertEqual(
            assignments,
            [(3, "zero_init"), (3, "supervised_init"), (4, "zero_init"), (4, "supervised_init")],
        )
        self.assertEqual(len(set(assignments)), 4)

    def test_awr_weights_are_finite_normalized_and_reward_sensitive(self) -> None:
        weights = episodic_awr_weights(
            torch.tensor([1.0, 0.0, 0.0, 1.0]), temperature=0.5, maximum_weight=20.0
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertTrue(torch.equal(weights[[0, 3]], weights[[0, 0]]))
        flat = episodic_awr_weights(
            torch.zeros(8), temperature=0.5, maximum_weight=20.0
        )
        self.assertTrue(torch.equal(flat, torch.ones(8)))

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

    def test_stage16_passes_early_or_continues_only_on_positive_trend(self) -> None:
        passed = decide_task_local_rl_node(self.spec, interaction_episodes=16, metrics=self._metrics())
        self.assertEqual(passed["status"], "rl_candidate_selected_for_fresh_gate")
        self.assertEqual(passed["selected_interaction_episodes"], 16)
        trend = decide_task_local_rl_node(
            self.spec, interaction_episodes=16, metrics=self._metrics(zero=(1, 0), supervised=(0, 0))
        )
        self.assertEqual(trend["status"], "continue_same_rl_trajectories_to_32_episodes")
        stopped = decide_task_local_rl_node(
            self.spec, interaction_episodes=16, metrics=self._metrics(zero=(0, 0), supervised=(0, 0))
        )
        self.assertEqual(stopped["status"], "task_local_rl_early_check_not_supported")

    def test_stage32_cannot_continue_or_authorize_writer(self) -> None:
        decision = decide_task_local_rl_node(
            self.spec, interaction_episodes=32, metrics=self._metrics(zero=(1, 0), supervised=(0, 0))
        )
        self.assertEqual(decision["status"], "task_local_rl_candidate_not_supported")
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

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
        self.assertIn('[[ "$stop_after_episodes" =~ ^(16|32)$ ]]', text)
        self.assertIn("--resume", text)
        self.assertIn("gpu_telemetry_", text)
        self.assertNotIn("ppo", text.lower())
        self.assertNotIn("writer", text.lower())

    def test_weighted_flow_loss_requires_mean_one_weights(self) -> None:
        loss = weighted_flow_loss(
            torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.5])
        )
        self.assertEqual(float(loss), 2.5)
        with self.assertRaises(Exception):
            weighted_flow_loss(torch.ones(2), torch.ones(2) * 2)


if __name__ == "__main__":
    unittest.main()
