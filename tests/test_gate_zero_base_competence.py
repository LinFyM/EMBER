from __future__ import annotations

import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_base_competence import (  # noqa: E402
    GateZeroBaseCompetenceError,
    assigned_competence_arms,
    decide_source_competence,
    load_source_competence_spec,
    resolve_competence_prompt,
    validate_source_competence_spec,
)


class GateZeroBaseCompetenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_source_competence.toml"
        cls.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        cls.phase0 = ROOT / "configs" / "phase0.toml"
        cls.spec = load_source_competence_spec(cls.path, cls.gate_zero, cls.phase0)

    def test_checked_in_surface_is_source_only_and_predeclared(self) -> None:
        self.assertEqual(self.spec["task_suite"], "libero_90")
        self.assertEqual(self.spec["task_ids"], [3, 4])
        self.assertEqual(self.spec["conditions"], ["correct", "swapped"])
        self.assertEqual(self.spec["official_init_state_indices"], list(range(8, 16)))
        self.assertEqual(self.spec["batch_size"], 8)
        self.assertTrue(self.spec["use_async_envs"])
        self.assertEqual(self.spec["parallel"]["preferred_world_size"], 4)
        self.assertEqual(
            self.spec["decision"]["correct_prompt_minimum_successes_per_task"], 2
        )
        self.assertFalse(self.spec["decision"]["gate_zero_decision_authorized"])
        self.assertFalse(self.spec["decision"]["writer_authorized"])

    def test_surface_rejects_any_non_source_or_extra_task(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["task_ids"] = [3, 61]
        with self.assertRaisesRegex(GateZeroBaseCompetenceError, "source tasks"):
            validate_source_competence_spec(changed, self.gate_zero, self.phase0)

    def test_surface_rejects_init_state_or_batch_drift(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["official_init_state_indices"] = list(range(0, 8))
        with self.assertRaisesRegex(GateZeroBaseCompetenceError, "init-state"):
            validate_source_competence_spec(changed, self.gate_zero, self.phase0)

    def test_arm_parallelism_covers_each_arm_once_for_one_two_or_four_ranks(self) -> None:
        expected = [(3, "correct"), (3, "swapped"), (4, "correct"), (4, "swapped")]
        for world_size in (1, 2, 4):
            assigned = [
                arm
                for rank in range(world_size)
                for arm in assigned_competence_arms(
                    self.spec, rank=rank, world_size=world_size
                )
            ]
            self.assertEqual(sorted(assigned), sorted(expected))
            self.assertEqual(len(assigned), len(set(assigned)))

    def test_correct_and_swapped_prompts_are_involutive(self) -> None:
        languages = {3: "put butter at back", 4: "put butter at front"}
        self.assertEqual(
            resolve_competence_prompt(self.spec, 3, "correct", languages), languages[3]
        )
        self.assertEqual(
            resolve_competence_prompt(self.spec, 3, "swapped", languages), languages[4]
        )
        self.assertEqual(
            resolve_competence_prompt(self.spec, 4, "swapped", languages), languages[3]
        )

    def test_decision_requires_two_correct_successes_per_task(self) -> None:
        arms = [
            {"task_id": 3, "condition": "correct", "successes": [True, True] + [False] * 6},
            {"task_id": 3, "condition": "swapped", "successes": [False] * 8},
            {"task_id": 4, "condition": "correct", "successes": [True, False] * 4},
            {"task_id": 4, "condition": "swapped", "successes": [False] * 8},
        ]
        decision = decide_source_competence(self.spec, arms, mechanics_valid=True)
        self.assertEqual(decision["status"], "source_competence_passed")
        self.assertTrue(decision["task_local_oracle_fit_authorized"])
        self.assertFalse(decision["gate_zero_authorized"])
        self.assertFalse(decision["writer_authorized"])

    def test_failure_authorizes_only_the_frozen_base_extension(self) -> None:
        arms = [
            {"task_id": 3, "condition": "correct", "successes": [True] + [False] * 7},
            {"task_id": 3, "condition": "swapped", "successes": [False] * 8},
            {"task_id": 4, "condition": "correct", "successes": [True, True] + [False] * 6},
            {"task_id": 4, "condition": "swapped", "successes": [False] * 8},
        ]
        decision = decide_source_competence(self.spec, arms, mechanics_valid=True)
        self.assertEqual(decision["status"], "source_competence_failed")
        self.assertEqual(decision["failure_class"], "source_base_acquisition")
        self.assertEqual(decision["bounded_recovery_max_steps"], 20000)
        self.assertFalse(decision["task_local_oracle_fit_authorized"])

    def test_mechanics_failure_blocks_scientific_interpretation(self) -> None:
        decision = decide_source_competence(self.spec, [], mechanics_valid=False)
        self.assertEqual(decision["status"], "stopped")
        self.assertEqual(decision["failure_class"], "implementation")
        self.assertFalse(decision["task_local_oracle_fit_authorized"])

    def test_shell_dry_run_uses_four_arm_parallel_ranks_and_offline_runtime(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_source_competence.sh"),
                "--gpus=4,5,6,7",
                "--output-dir=/tmp/ember-source-competence",
                "--dry-run",
            ],
            cwd=ROOT,
            env={**os.environ, "EMBER_PYTHON": sys.executable},
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=4\\,5\\,6\\,7", completed.stdout)
        self.assertIn("--nproc-per-node=4", completed.stdout)
        self.assertIn("-m ember.gate_zero_base_competence", completed.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", completed.stdout)
        launcher = (ROOT / "scripts" / "run_gate_zero_source_competence.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("trap 'handle_signal 130' INT", launcher)
        self.assertIn("trap 'handle_signal 143' TERM", launcher)
        self.assertNotIn(" EXIT", launcher)


if __name__ == "__main__":
    unittest.main()
