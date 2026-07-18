from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_model_probe import (  # noqa: E402
    GateZeroModelProbeError,
    parameter_summary,
    validate_output_destination,
)


class GateZeroModelProbeTest(unittest.TestCase):
    def test_parameter_summary_counts_only_trainable_finite_parameters(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        model[1].requires_grad_(False)

        summary = parameter_summary(model)

        self.assertEqual(summary["trainable_parameters"], 8)
        self.assertEqual(summary["trainable_tensors"], 2)
        self.assertEqual(summary["total_parameters"], 11)

    def test_completed_output_is_never_overwritten(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "probe_result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(GateZeroModelProbeError, "completed"):
                validate_output_destination(output)


if __name__ == "__main__":
    unittest.main()
