from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_runtime import (  # noqa: E402
    GateZeroRuntimeError,
    build_lora_config,
    deterministic_flow_inputs,
    inspect_lora_targets,
    load_source_normalization,
)


class GateZeroRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_source_normalization_is_hash_and_authority_bound(self) -> None:
        path = self.root / "normalization.json"
        payload = {
            "authority": {
                "episode_bounds_inclusive": [8, 27],
                "split": "source",
                "task_indices": [3, 4],
            },
            "observation.state": {
                "count": 4,
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "min": [-1.0] * 8,
                "max": [1.0] * 8,
                "q01": [-0.9] * 8,
                "q99": [0.9] * 8,
            },
            "action": {
                "count": 4,
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
                "q01": [-0.9] * 7,
                "q99": [0.9] * 7,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        stats = load_source_normalization(
            path,
            expected_sha256=digest,
            expected_task_ids=[3, 4],
            expected_count=4,
        )

        self.assertEqual(tuple(stats["observation.state"]["mean"].shape), (8,))
        self.assertEqual(tuple(stats["action"]["std"].shape), (7,))
        self.assertEqual(stats["action"]["mean"].dtype, torch.float32)
        with self.assertRaisesRegex(GateZeroRuntimeError, "task authority"):
            load_source_normalization(
                path,
                expected_sha256=digest,
                expected_task_ids=[3],
                expected_count=4,
            )

    def test_target_inspection_uses_physical_weight_shapes(self) -> None:
        path = self.root / "model.safetensors"
        targets = ["model.layer.q_proj", "model.layer.v_proj"]
        save_file(
            {
                "model.layer.q_proj.weight": torch.zeros(6, 4),
                "model.layer.v_proj.weight": torch.zeros(2, 4),
                "ignored.weight": torch.zeros(1),
            },
            path,
        )

        result = inspect_lora_targets(path, targets, rank=2)

        self.assertEqual(result["trainable_parameters"], 2 * (6 + 4) + 2 * (2 + 4))
        self.assertEqual(result["targets"][0]["shape"], [6, 4])

    def test_fixed_flow_inputs_are_row_keyed_and_batch_order_invariant(self) -> None:
        first_noise, first_time = deterministic_flow_inputs(
            ["task3/demo40/frame0", "task3/demo40/frame1"],
            action_shape=(3, 7),
            noise_seed=10,
            time_seed=11,
            device=torch.device("cpu"),
        )
        reversed_noise, reversed_time = deterministic_flow_inputs(
            ["task3/demo40/frame1", "task3/demo40/frame0"],
            action_shape=(3, 7),
            noise_seed=10,
            time_seed=11,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(first_noise[0], reversed_noise[1], rtol=0, atol=0)
        torch.testing.assert_close(first_noise[1], reversed_noise[0], rtol=0, atol=0)
        torch.testing.assert_close(first_time[0], reversed_time[1], rtol=0, atol=0)
        self.assertTrue(torch.all((first_time >= 0.001) & (first_time <= 1.0)))

    def test_lora_config_preserves_full_targets_and_zero_functional_init_choice(self) -> None:
        targets = ["model.layer.q_proj", "model.layer.v_proj"]
        config = build_lora_config(
            targets=targets,
            rank=8,
            alpha=8,
            dropout=0.0,
            init_lora_weights="orthogonal",
            base_revision="c" * 40,
        )

        self.assertEqual(set(config.target_modules), set(targets))
        self.assertEqual(config.init_lora_weights, "orthogonal")
        self.assertEqual(config.modules_to_save, [])
        self.assertEqual(config.revision, "c" * 40)


if __name__ == "__main__":
    unittest.main()
