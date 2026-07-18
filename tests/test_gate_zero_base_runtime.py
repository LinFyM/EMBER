from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_base_runtime import (  # noqa: E402
    build_base_optimizer,
    build_base_scheduler,
    capture_trainable_state,
    gradient_accumulation_steps,
    optimizer_state_summary,
    restore_trainable_state,
    training_row_keys,
)


class GateZeroBaseRuntimeTest(unittest.TestCase):
    def test_effective_batch_is_exact(self) -> None:
        self.assertEqual(gradient_accumulation_steps(64, 8), 8)
        self.assertEqual(gradient_accumulation_steps(64, 64), 1)
        self.assertEqual(gradient_accumulation_steps(64, 32, world_size=2), 1)
        self.assertEqual(gradient_accumulation_steps(64, 16, world_size=4), 1)
        with self.assertRaisesRegex(ValueError, "divide"):
            gradient_accumulation_steps(64, 24)

    def test_optimizer_and_upstream_scheduler_match_frozen_curve(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        spec = {
            "base_fit": {
                "learning_rate": 1e-4,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "weight_decay": 1e-10,
                "warmup_steps": 1000,
                "decay_steps": 10000,
                "decay_learning_rate": 2.5e-6,
                "steps": 10000,
            }
        }
        optimizer = build_base_optimizer([parameter], spec)
        scheduler = build_base_scheduler(optimizer, spec)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4 / 1001)
        for completed_step in range(1, 10001):
            optimizer.step()
            scheduler.step()
            if completed_step == 1000:
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 9.761400516938874e-5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2.5e-6)

    def test_flow_row_keys_use_absolute_effective_batch_slots(self) -> None:
        def raw(start: int, size: int) -> dict[str, torch.Tensor]:
            values = torch.arange(start, start + size)
            return {
                "task_id": torch.full((size,), 3),
                "demo_index": torch.full((size,), 8),
                "frame_index": values,
            }

        keys_8 = [
            key
            for accumulation in range(8)
            for key in training_row_keys(
                raw(accumulation * 8, 8),
                optimizer_step=4,
                effective_batch_start_slot=accumulation * 8,
            )
        ]
        keys_16 = [
            key
            for accumulation in range(4)
            for key in training_row_keys(
                raw(accumulation * 16, 16),
                optimizer_step=4,
                effective_batch_start_slot=accumulation * 16,
            )
        ]

        self.assertEqual(keys_8, keys_16)

    def test_trainable_snapshot_restores_only_trainable_parameters(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        model[1].requires_grad_(False)
        frozen_before = {name: value.detach().clone() for name, value in model[1].named_parameters()}
        snapshot = capture_trainable_state(model)
        with torch.no_grad():
            for value in model.parameters():
                value.add_(1)

        restore_trainable_state(model, snapshot)

        for name, value in model[0].named_parameters():
            torch.testing.assert_close(value, snapshot[f"0.{name}"], rtol=0, atol=0)
        for name, value in model[1].named_parameters():
            torch.testing.assert_close(value, frozen_before[name] + 1, rtol=0, atol=0)

    def test_optimizer_state_summary_reports_actual_tensor_dtypes(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0, dtype=torch.float32),
            "exp_avg": torch.ones(3, dtype=torch.bfloat16),
            "exp_avg_sq": torch.ones(3, dtype=torch.bfloat16),
        }

        summary = optimizer_state_summary(optimizer)

        self.assertEqual(summary["parameter_dtype_elements"], {"torch.bfloat16": 3})
        self.assertEqual(
            summary["state_tensor_elements_by_dtype"],
            {"torch.bfloat16": 6, "torch.float32": 1},
        )
        self.assertEqual(summary["state_tensor_elements_by_key"]["exp_avg"], 3)


if __name__ == "__main__":
    unittest.main()
