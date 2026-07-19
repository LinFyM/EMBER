from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from ember.gate_zero_support.base_difficulty_audit import (
    assigned_tasks,
    load_difficulty_audit_spec,
    set_physical_init_state_ids,
)


ROOT = Path(__file__).resolve().parents[1]


class GateZeroBaseDifficultyAuditTest(unittest.TestCase):
    def test_contract_is_source_only_and_matches_repaired_evidence(self) -> None:
        spec, evidence, _split = load_difficulty_audit_spec(
            ROOT / "configs" / "gate_zero_base_difficulty_audit.toml",
            ROOT / "configs" / "gate_zero_evidence_repair.toml",
            ROOT / "configs" / "libero90_split_reseal.json",
        )
        self.assertEqual(spec["task_ids"], evidence["confirmation_selection"]["candidate_task_ids"])
        self.assertEqual(spec["episodes_per_task"], 32)
        self.assertEqual(spec["batch_size"], 8)
        self.assertEqual(spec["policy_rng_seeds"], [2026071971, 2026071972, 2026071973, 2026071974])
        self.assertEqual(spec["execution_horizon"], 16)
        self.assertFalse(spec["validation_numeric_access"])
        self.assertFalse(spec["held_numeric_access"])
        self.assertFalse(spec["locked_numeric_access"])

    def test_world4_assignment_is_complete_and_nonduplicated(self) -> None:
        spec, _evidence, _split = load_difficulty_audit_spec(
            ROOT / "configs" / "gate_zero_base_difficulty_audit.toml",
            ROOT / "configs" / "gate_zero_evidence_repair.toml",
            ROOT / "configs" / "libero90_split_reseal.json",
        )
        assigned = [
            task_id
            for rank in range(4)
            for task_id in assigned_tasks(spec, rank=rank, world_size=4)
        ]
        self.assertEqual(sorted(assigned), spec["task_ids"])
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_physical_state_setter_binds_before_and_after_ids(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.values = [0, 1]

            def set_attr(self, name, values) -> None:
                self.assert_name = name
                self.values = list(values)

            def call(self, name):
                self.assert_call = name
                return self.values

        env = FakeEnv()
        self.assertEqual(set_physical_init_state_ids(env, [11, 37]), [11, 37])
        self.assertEqual(env.assert_name, "init_state_id")
        self.assertEqual(env.assert_call, "init_state_id")

    def test_physical_state_setter_supports_lerobot_lazy_vector_wrapper(self) -> None:
        class Actual:
            def __init__(self) -> None:
                self.values = [0, 1]

            def set_attr(self, name, values) -> None:
                self.values = list(values)

            def call(self, name):
                return self.values

            def get_attr(self, name):
                return tuple(self.values)

        class Lazy:
            def __init__(self) -> None:
                self._env = None

            def _ensure(self) -> None:
                self._env = self._env or Actual()

            def get_attr(self, name):
                self._ensure()
                return self._env.get_attr(name)

            def call(self, name):
                self._ensure()
                return self._env.call(name)

        class AuditProxy:
            def __init__(self) -> None:
                self._env = Lazy()

            def __getattr__(self, name):
                return getattr(self._env, name)

        env = AuditProxy()
        self.assertEqual(set_physical_init_state_ids(env, [11, 37]), [11, 37])

    def test_launcher_dry_run_uses_one_canonical_module(self) -> None:
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / "run_gate_zero_base_difficulty_audit.sh"),
                "--gpus=4,5,6,7",
                "--output-dir=/tmp/ember-difficulty-audit-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("-m ember.gate_zero_support.base_difficulty_audit", completed.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=4\\,5\\,6\\,7", completed.stdout)


if __name__ == "__main__":
    unittest.main()
