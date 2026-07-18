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


class GateZeroDdpIntegrationTest(unittest.TestCase):
    def test_two_and_four_rank_gradient_and_flow_match_single_global_batch(self) -> None:
        reference = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            reference.weight.copy_(torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.7]]))
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        inputs = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3) / 100.0
        targets = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2) / 50.0
        torch.nn.functional.mse_loss(reference(inputs), targets).backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as temporary:
            for world_size in (2, 4):
                output = Path(temporary) / f"world_{world_size}.json"
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT / "src")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={world_size}",
                        str(ROOT / "tests" / "helpers" / "gate_zero_ddp_worker.py"),
                        "--config",
                        str(ROOT / "configs" / "gate_zero_training_topology.toml"),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                result = json.loads(output.read_text(encoding="utf-8"))
                torch.testing.assert_close(
                    torch.tensor(result["weight"]),
                    reference.weight.detach(),
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertTrue(result["flow_noise_exact"])
                self.assertTrue(result["flow_time_exact"])
                self.assertTrue(result["flow_input_sha256_exact"])
                self.assertTrue(result["ddp_static_graph"])
                self.assertEqual(result["rank"], 0)

    def test_two_and_four_rank_same_topology_checkpoint_resume_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for world_size in (2, 4):
                root = Path(temporary) / f"world_{world_size}"
                output = root / "result.json"
                checkpoint = root / "checkpoints" / "000001"
                root.mkdir(parents=True)
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT / "src")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={world_size}",
                        str(ROOT / "tests" / "helpers" / "gate_zero_ddp_resume_worker.py"),
                        "--config",
                        str(ROOT / "configs" / "gate_zero_training_topology.toml"),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(result["all_model_exact"])
                self.assertTrue(result["all_lr_exact"])
                self.assertTrue(result["all_rng_exact"])
                self.assertTrue(result["reference_static_graph"])
                self.assertTrue(result["resumed_static_graph"])
                self.assertEqual(result["checkpoint_schema_version"], 3)
                self.assertEqual(result["checkpoint_world_size"], world_size)
                self.assertEqual(len(result["distributed_rng_files"]), world_size)
                self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
