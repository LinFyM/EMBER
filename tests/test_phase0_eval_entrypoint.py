from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase0_eval.sh"


class Phase0EvalEntrypointTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            environment = os.environ.copy()
            environment.update(
                {
                    "EMBER_ASSET_ROOT": str(temporary_root / "assets"),
                    "EMBER_DATA_ROOT": str(temporary_root / "data"),
                    "HF_HOME": str(temporary_root / "cache"),
                    "LIBERO_CONFIG_PATH": str(temporary_root / "libero-config"),
                }
            )
            return subprocess.run(
                ["bash", str(SCRIPT), *args],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_dry_run_keeps_offline_task_and_seed_contract(self) -> None:
        result = self._run(
            "--gpu=5",
            "--task-suite=libero_spatial",
            "--task-ids=[0]",
            "--episodes=8",
            "--batch-size=8",
            "--seed=1000",
            "--output-dir=/tmp/ember-phase0-test-output",
            "--dry-run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=5", result.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", result.stdout)
        self.assertIn("TRANSFORMERS_OFFLINE=1", result.stdout)
        self.assertIn("--env.task=libero_spatial", result.stdout)
        self.assertIn("--env.task_ids=\\[0\\]", result.stdout)
        self.assertIn("--eval.n_episodes=8", result.stdout)
        self.assertIn("--eval.batch_size=8", result.stdout)
        self.assertIn("--seed=1000", result.stdout)
        self.assertNotIn("--env.seed", result.stdout)
        self.assertIn("31d453f7edd78c839a8bbc39744a292686daf0de", result.stdout)

    def test_rejects_rollouts_that_batch_would_discard(self) -> None:
        result = self._run(
            "--gpu=5",
            "--episodes=7",
            "--batch-size=4",
            "--output-dir=/tmp/ember-phase0-test-output",
            "--dry-run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible", result.stderr)

    def test_rejects_multi_gpu_value_for_single_process_entrypoint(self) -> None:
        result = self._run(
            "--gpu=4,5",
            "--output-dir=/tmp/ember-phase0-test-output",
            "--dry-run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one physical GPU index", result.stderr)

    def test_dry_run_can_select_async_vector_environments(self) -> None:
        result = self._run(
            "--gpu=5",
            "--episodes=8",
            "--batch-size=8",
            "--async-envs=true",
            "--output-dir=/tmp/ember-phase0-test-output",
            "--dry-run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--eval.use_async_envs=true", result.stdout)

    def test_rejects_invalid_async_value(self) -> None:
        result = self._run(
            "--gpu=5",
            "--async-envs=maybe",
            "--output-dir=/tmp/ember-phase0-test-output",
            "--dry-run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--async-envs", result.stderr)


if __name__ == "__main__":
    unittest.main()
