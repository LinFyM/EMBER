from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_task_local_rl.contract import (  # noqa: E402
    GateZeroTaskLocalRLContractError,
    assigned_task_local_rl_arm,
    decide_task_local_rl_node,
    load_task_local_rl_spec,
    validate_task_local_rl_spec,
)
from ember.gate_zero_task_local_rl.runtime import (  # noqa: E402
    AnchorRecordingEnvPreprocessor,
    build_balanced_replay_batch,
    scoped_policy_execution_horizon,
    validated_flow_action_shape,
    validate_training_reset_events,
)
from ember.gate_zero_task_local_rl.temporal_credit import (  # noqa: E402
    TemporalCritic,
    _actor_update_enabled,
    _flow_losses_microbatched,
    _select_real_camera_inputs,
    calculate_masked_gae,
    clipped_flow_ppo_loss,
)
from ember.gate_zero_oracle_artifacts import (  # noqa: E402
    load_recovery_artifact,
    save_recovery_artifact,
)
from ember.gate_zero_oracle_session import capture_trainable_state  # noqa: E402


class GateZeroTaskLocalRLRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_task_local_rl_critic_warmup.toml"
        cls.horizon_path = ROOT / "configs" / "gate_zero_task_local_rl_horizon_credit.toml"
        cls.coverage_path = ROOT / "configs" / "gate_zero_task_local_rl_horizon_coverage.toml"
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
        self.assertEqual(
            self.spec["training_interaction"]["interaction_episode_nodes"],
            [8, 16, 24, 32],
        )
        self.assertEqual(self.spec["exploration"]["standard_deviation"], [0.0] * 7)
        self.assertEqual(
            self.spec["algorithm"]["name"],
            "chunk_level_flow_ppo_with_task_local_critic_warmup_and_gae",
        )
        self.assertTrue(self.spec["algorithm"]["not_full_fpo_plus"])
        self.assertEqual(self.spec["algorithm"]["flow_samples_per_transition"], 8)
        self.assertEqual(self.spec["algorithm"]["critic"], "task_local_frozen_feature_mlp")
        self.assertEqual(self.spec["algorithm"]["discount"], 0.99)
        self.assertEqual(self.spec["algorithm"]["gae_lambda"], 0.99)
        self.assertEqual(self.spec["algorithm"]["critic_only_rounds"], 1)

    def test_horizon_credit_contract_changes_only_training_credit_resolution(self) -> None:
        horizon = load_task_local_rl_spec(
            self.horizon_path,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            fit_path=self.fit,
            headroom_path=self.headroom,
            diagnostic_path=self.diagnostic,
        )
        self.assertEqual(horizon["task_ids"], [3, 4])
        self.assertEqual(horizon["reported_arms"], self.spec["reported_arms"])
        self.assertEqual(horizon["lora"], self.spec["lora"])
        self.assertEqual(horizon["algorithm"]["execution_horizon"], 16)
        self.assertEqual(horizon["algorithm"]["action_chunk_size"], 50)
        self.assertEqual(horizon["algorithm"]["anchors_per_episode"], 25)
        self.assertEqual(horizon["algorithm"]["effective_replay_batch_size"], 200)
        for key in (
            "init_state_indices",
            "batch_size",
            "seed_start",
            "warmup_seed_start",
            "policy_rng_seed",
            "frozen_base_successes_by_task",
            "supervised_lora_successes_by_task",
        ):
            self.assertEqual(
                horizon["development_evaluation"][key],
                self.spec["development_evaluation"][key],
            )
        self.assertEqual(
            horizon["development_evaluation"]["evaluate_after_interaction_episodes"],
            [8, 16],
        )
        self.assertEqual(
            horizon["candidate_decision"], self.spec["candidate_decision"]
        )
        self.assertEqual(
            horizon["training_interaction"]["interaction_episode_nodes"], [8, 16]
        )

    def test_four_rank_assignment_has_no_duplicate_or_idle_arm(self) -> None:
        assignments = [assigned_task_local_rl_arm(rank=i, world_size=4, spec=self.spec) for i in range(4)]
        self.assertEqual(
            assignments,
            [(3, "zero_init"), (3, "supervised_init"), (4, "zero_init"), (4, "supervised_init")],
        )
        self.assertEqual(len(set(assignments)), 4)

    def _metrics(self, *, zero=(0, 0), supervised=(0, 0), drift=0.01) -> dict:
        return {
            "mechanics_valid": True,
            "temporal_credit_healthy": True,
            "maximum_saturation_fraction": 0.0,
            "nonfinite_count": 0,
            "action_drift_by_arm": {"zero_init_rl": drift, "supervised_init_rl": drift},
            "critic_warmup_actor_state_unchanged": True,
            "paired_net_wins_by_arm": {
                "zero_init_rl": list(zero),
                "supervised_init_rl": list(supervised),
            },
        }

    def test_critic_warmup_and_result_blind_four_node_decisions(self) -> None:
        continued = decide_task_local_rl_node(
            self.spec,
            interaction_episodes=8,
            metrics=self._metrics(),
        )
        self.assertEqual(continued["status"], "critic_warmup_complete_continue_to_16")
        identity_failed = self._metrics(zero=(1, 0))
        self.assertEqual(
            decide_task_local_rl_node(
                self.spec, interaction_episodes=8, metrics=identity_failed
            )["status"],
            "task_local_rl_mechanical_or_safeguard_failure",
        )
        unhealthy = self._metrics()
        unhealthy["temporal_credit_healthy"] = False
        stopped_early = decide_task_local_rl_node(
            self.spec, interaction_episodes=8, metrics=unhealthy
        )
        self.assertEqual(stopped_early["status"], "task_local_rl_mechanical_or_safeguard_failure")
        stage16 = decide_task_local_rl_node(
            self.spec, interaction_episodes=16, metrics=self._metrics()
        )
        self.assertEqual(stage16["status"], "critic_warmup_recovery_continue_to_24")
        stopped_at24 = decide_task_local_rl_node(
            self.spec, interaction_episodes=24, metrics=self._metrics()
        )
        self.assertEqual(stopped_at24["status"], "task_local_rl_early_check_not_supported")
        trended = decide_task_local_rl_node(
            self.spec,
            interaction_episodes=24,
            metrics=self._metrics(zero=(1, 0)),
        )
        self.assertEqual(trended["status"], "critic_warmup_recovery_continue_to_32")
        passed = decide_task_local_rl_node(
            self.spec,
            interaction_episodes=32,
            metrics=self._metrics(zero=(2, 1)),
        )
        self.assertEqual(passed["status"], "rl_candidate_selected_for_fresh_gate")
        self.assertEqual(passed["selected_interaction_episodes"], 32)
        stopped = decide_task_local_rl_node(
            self.spec, interaction_episodes=32, metrics=self._metrics()
        )
        self.assertEqual(stopped["status"], "task_local_rl_early_check_not_supported")

    def test_horizon_credit_stops_after_one_actor_round_without_gain(self) -> None:
        horizon = load_task_local_rl_spec(
            self.horizon_path,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            fit_path=self.fit,
            headroom_path=self.headroom,
            diagnostic_path=self.diagnostic,
        )
        stage8 = decide_task_local_rl_node(
            horizon, interaction_episodes=8, metrics=self._metrics()
        )
        self.assertEqual(stage8["status"], "horizon_credit_warmup_complete_continue_to_16")
        stage16 = decide_task_local_rl_node(
            horizon, interaction_episodes=16, metrics=self._metrics()
        )
        self.assertEqual(stage16["status"], "task_local_rl_early_check_not_supported")
        passing = decide_task_local_rl_node(
            horizon, interaction_episodes=16, metrics=self._metrics(zero=(2, 1))
        )
        self.assertEqual(passing["status"], "rl_candidate_selected_for_fresh_gate")
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            decide_task_local_rl_node(
                horizon, interaction_episodes=24, metrics=self._metrics(zero=(2, 1))
            )

    def test_horizon_coverage_recovery_adds_only_disjoint_source_rounds(self) -> None:
        coverage = load_task_local_rl_spec(
            self.coverage_path,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            fit_path=self.fit,
            headroom_path=self.headroom,
            diagnostic_path=self.diagnostic,
        )
        horizon = load_task_local_rl_spec(
            self.horizon_path,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            fit_path=self.fit,
            headroom_path=self.headroom,
            diagnostic_path=self.diagnostic,
        )
        self.assertEqual(coverage["lora"], horizon["lora"])
        self.assertEqual(coverage["algorithm"], horizon["algorithm"])
        self.assertEqual(coverage["candidate_decision"], horizon["candidate_decision"])
        self.assertEqual(coverage["development_evaluation"]["init_state_indices"], list(range(40, 48)))
        self.assertEqual(
            coverage["training_interaction"]["train_init_state_indices_by_round"],
            [list(range(start, start + 8)) for start in (8, 16, 24, 32)],
        )
        self.assertEqual(
            coverage["training_interaction"]["interaction_episode_nodes"],
            [8, 16, 24, 32],
        )
        stage8 = decide_task_local_rl_node(
            coverage, interaction_episodes=8, metrics=self._metrics()
        )
        self.assertEqual(stage8["status"], "horizon_coverage_warmup_complete_continue_to_16")
        stage16 = decide_task_local_rl_node(
            coverage,
            interaction_episodes=16,
            metrics=self._metrics(zero=(-2, 1), supervised=(0, -1)),
        )
        self.assertEqual(stage16["status"], "horizon_coverage_recovery_continue_to_24")
        stopped = decide_task_local_rl_node(
            coverage, interaction_episodes=24, metrics=self._metrics()
        )
        self.assertEqual(stopped["status"], "task_local_rl_early_check_not_supported")
        trended = decide_task_local_rl_node(
            coverage, interaction_episodes=24, metrics=self._metrics(zero=(1, 0))
        )
        self.assertEqual(trended["status"], "horizon_coverage_recovery_continue_to_32")
        terminal = decide_task_local_rl_node(
            coverage, interaction_episodes=32, metrics=self._metrics()
        )
        self.assertEqual(terminal["status"], "task_local_rl_early_check_not_supported")
        changed = copy.deepcopy(coverage)
        changed["coverage_evidence"]["horizon_support_replay_result_sha256"] = "0" * 64
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )

    def test_stage40_is_not_an_active_decision_node(self) -> None:
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            decide_task_local_rl_node(
                self.spec, interaction_episodes=40, metrics=self._metrics(zero=(1, 0))
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
        changed["predecessor_evidence"]["temporal_credit_result_sha256"] = "0" * 64
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
        changed["algorithm"]["flow_samples_per_transition"] = 1
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
        changed["authority"]["previous_temporal_result_sha256"] = "0" * 64
        with self.assertRaises(GateZeroTaskLocalRLContractError):
            validate_task_local_rl_spec(
                changed,
                gate_zero_path=self.gate_zero,
                phase0_path=self.phase0,
                fit_path=self.fit,
                headroom_path=self.headroom,
                diagnostic_path=self.diagnostic,
            )

    def test_replay_builder_preserves_temporal_order_and_masks_padded_suffix(self) -> None:
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
        self.assertEqual(replay["transition_valid"].reshape(2, 8).sum(dim=1).tolist(), [1, 2])
        self.assertEqual(replay["transition_reward"].reshape(2, 8)[0].tolist(), [1.0] + [0.0] * 7)
        self.assertTrue(replay["transition_done"].reshape(2, 8)[0, 0])
        self.assertTrue(replay["transition_done"].reshape(2, 8)[1, 1])
        self.assertEqual(replay["trajectory_shape"], [2, 8])
        self.assertEqual(len(replay["row_keys"]), 16)
        self.assertEqual(len(set(replay["row_keys"])), 16)
        self.assertTrue(replay["action_is_pad"][0, 21:].all())

    def test_horizon_resolved_replay_masks_unexecuted_model_chunk_suffix(self) -> None:
        batch_size = 1
        anchors = [
            {
                "step": step,
                "observation.images.camera1": torch.zeros(
                    batch_size, 3, 4, 4, dtype=torch.uint8
                ),
                "observation.images.camera2": torch.zeros(
                    batch_size, 3, 4, 4, dtype=torch.uint8
                ),
                "observation.state": torch.zeros(batch_size, 8),
                "task": ["task"],
            }
            for step in (0, 16)
        ]
        actions = torch.arange(40 * 7, dtype=torch.float32).reshape(1, 40, 7)
        done = torch.zeros(1, 40, dtype=torch.bool)
        done[:, 39] = True
        success = torch.zeros(1, 40, dtype=torch.bool)
        success[:, 20] = True
        replay = build_balanced_replay_batch(
            anchors=anchors,
            rollout={"action": actions, "done": done, "success": success},
            seeds=[6200],
            task_id=3,
            action_chunk_size=50,
            execution_horizon=16,
            anchors_per_episode=2,
        )
        self.assertEqual(replay["action"].shape, (2, 50, 7))
        self.assertTrue(torch.equal(replay["action"][0, :16], actions[0, :16]))
        self.assertTrue(replay["action_is_pad"][0, 16:].all())
        self.assertFalse(replay["action_is_pad"][0, :16].any())
        self.assertEqual(replay["transition_reward"].tolist(), [0.0, 1.0])

    def test_policy_execution_horizon_is_scoped_and_restored(self) -> None:
        class Policy:
            def __init__(self) -> None:
                self.config = SimpleNamespace(n_action_steps=50, chunk_size=50)
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        policy = Policy()
        with scoped_policy_execution_horizon(
            policy, execution_horizon=16, expected_model_chunk_size=50
        ):
            self.assertEqual(policy.config.n_action_steps, 16)
            self.assertEqual(policy.config.chunk_size, 50)
            self.assertEqual(policy.reset_calls, 1)
        self.assertEqual(policy.config.n_action_steps, 50)
        self.assertEqual(policy.config.chunk_size, 50)
        self.assertEqual(policy.reset_calls, 2)

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
        self.assertIn("gate_zero_task_local_rl_critic_warmup.toml", text)
        self.assertIn("--nproc-per-node=4", text)
        self.assertIn('spec["training_interaction"]["interaction_episode_nodes"]', text)
        self.assertIn("first_interaction_node", text)
        self.assertIn("--resume", text)
        self.assertIn("gpu_telemetry_", text)
        self.assertIn("horizon_support_replay_result_sha256", text)
        self.assertNotIn("writer", text.lower())

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


class GateZeroTemporalCreditRecoveryTest(unittest.TestCase):
    def test_flow_loss_capture_is_ordered_and_memory_bounded(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def forward(self, batch, *, noise, time, reduction):
                self.batch_sizes.append(len(batch["action"]))
                return batch["action"][:, 0, 0] + noise[:, 0, 0] * 0 + time[:, 0] * 0, {}

        model = FakeModel()
        batch = {"action": torch.arange(11, dtype=torch.float32).reshape(11, 1, 1)}
        noises = [torch.zeros(11, 1, 1), torch.ones(11, 1, 1)]
        times = [torch.zeros(11, 1), torch.ones(11, 1)]
        losses = _flow_losses_microbatched(
            model,
            batch,
            noises,
            times,
            torch.arange(11),
            microbatch_size=4,
        )
        self.assertEqual(losses.shape, (11, 2))
        self.assertTrue(torch.equal(losses[:, 0], torch.arange(11)))
        self.assertTrue(torch.equal(losses[:, 1], torch.arange(11)))
        self.assertLessEqual(max(model.batch_sizes), 4)

    def test_actor_updates_start_only_after_frozen_critic_warmup(self) -> None:
        algorithm = {"critic_only_rounds": 1}
        self.assertFalse(_actor_update_enabled(algorithm, round_index=0))
        self.assertTrue(_actor_update_enabled(algorithm, round_index=1))

    def test_masked_gae_propagates_terminal_reward_only_through_valid_time(self) -> None:
        rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        values = torch.zeros_like(rewards)
        dones = torch.tensor([[False, False, True, True], [False, True, True, True]])
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        advantages, returns = calculate_masked_gae(
            rewards,
            values,
            dones,
            valid,
            discount=1.0,
            gae_lambda=1.0,
        )
        self.assertTrue(torch.equal(advantages[0], torch.tensor([1.0, 1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(advantages[1], torch.zeros(4)))
        self.assertTrue(torch.equal(returns, advantages))

    def test_chunk_flow_ppo_averages_matched_samples_and_masks_padding(self) -> None:
        current = torch.tensor([[1.0, 1.0], [1.0, 1.0], [9.0, 9.0]], requires_grad=True)
        old = torch.ones_like(current)
        advantages = torch.tensor([1.0, -1.0, 100.0])
        valid = torch.tensor([True, True, False])
        loss, metrics = clipped_flow_ppo_loss(
            current,
            old,
            advantages,
            valid,
            ratio_clip=0.01,
            log_ratio_clamp=5.0,
        )
        loss.backward()
        self.assertGreater(float(current.grad[0].mean()), 0.0)
        self.assertLess(float(current.grad[1].mean()), 0.0)
        self.assertTrue(torch.equal(current.grad[2], torch.zeros(2)))
        self.assertEqual(metrics["valid_transitions"], 2)
        self.assertAlmostEqual(metrics["ratio_mean"], 1.0, places=6)

    def test_task_local_critic_has_zero_initial_value_and_expected_shape(self) -> None:
        critic = TemporalCritic(input_dim=17, hidden_dims=(12, 7))
        values = critic(torch.randn(5, 17))
        self.assertEqual(values.shape, (5,))
        self.assertTrue(torch.equal(values, torch.zeros(5)))

    def test_frozen_critic_excludes_declared_empty_camera(self) -> None:
        images = [torch.full((2, 1), value) for value in (1.0, 2.0, 0.0)]
        masks = [
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        ]
        selected_images, selected_masks = _select_real_camera_inputs(
            images, masks, empty_cameras=1
        )
        self.assertEqual(len(selected_images), 2)
        self.assertEqual(len(selected_masks), 2)
        self.assertTrue(torch.equal(selected_images[1], images[1]))

    def test_atomic_recovery_restores_auxiliary_critic_and_optimizer(self) -> None:
        model = torch.nn.Linear(3, 2, bias=False)
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        critic = TemporalCritic(input_dim=4, hidden_dims=(6, 5))
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)
        with torch.no_grad():
            critic.mlp[-1].weight.fill_(0.25)
        expected = {name: value.detach().clone() for name, value in critic.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = save_recovery_artifact(
                root,
                variant="temporal_credit",
                task_id=3,
                step=8,
                trainable_state=capture_trainable_state(model),
                optimizer=actor_optimizer,
                auxiliary_module=critic,
                auxiliary_optimizer=critic_optimizer,
                authorities={"contract": "frozen"},
            )
            with torch.no_grad():
                for value in critic.parameters():
                    value.zero_()
            restored = load_recovery_artifact(
                recovery,
                model=model,
                optimizer=actor_optimizer,
                auxiliary_module=critic,
                auxiliary_optimizer=critic_optimizer,
                expected={"step": 8, "authorities": {"contract": "frozen"}},
            )
        self.assertEqual(restored, 8)
        for name, value in critic.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)


if __name__ == "__main__":
    unittest.main()
