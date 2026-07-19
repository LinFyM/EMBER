from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_oracle_contract import load_oracle_fit_spec  # noqa: E402
from ember.gate_zero_oracle_metrics import select_action_mse_candidate  # noqa: E402
from ember.gate_zero_oracle_session import (  # noqa: E402
    full_sampler_generated_action_mse,
)
from ember.gate_zero_support.action_aligned_contract import (  # noqa: E402
    GateZeroActionAlignedContractError,
    load_action_aligned_acquisition_spec,
)


class _ToyActionOwner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(num_steps=10, max_action_dim=3)
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def _get_action_chunk(
        self, batch: dict[str, torch.Tensor], *, noise: torch.Tensor
    ) -> torch.Tensor:
        return self.scale * batch["features"] + 0.0 * noise.sum()


class _ToyPeftWrapper(torch.nn.Module):
    def __init__(self, owner: _ToyActionOwner) -> None:
        super().__init__()
        self.owner = owner

    def get_base_model(self) -> _ToyActionOwner:
        return self.owner


class GateZeroActionAlignedAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "configs" / "gate_zero_action_aligned_lora_acquisition.toml"
        self.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0 = ROOT / "configs" / "phase0.toml"
        self.competence = ROOT / "configs" / "gate_zero_source_competence.toml"

    def load(self):
        return load_action_aligned_acquisition_spec(
            self.config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )

    def test_checked_in_contract_freezes_one_short_action_aligned_recovery(self) -> None:
        spec = self.load()
        variant = spec["fit"]["action_aligned_official_default_r32"]

        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["fit"]["support_episode_bounds"], [0, 39])
        self.assertEqual(spec["selection"]["query_episode_bounds"], [40, 45])
        self.assertEqual(spec["fit"]["candidate_steps"], [0, 1, 5, 10, 25, 50, 100, 200])
        self.assertEqual(variant["training_objective"], "full_sampler_generated_action_mse")
        self.assertEqual(variant["action_loss_sampler_steps"], 10)
        self.assertEqual(variant["action_loss_noise_dimension"], 32)
        self.assertEqual(variant["generated_action_dimension"], 7)
        self.assertEqual(variant["rank"], 32)
        self.assertEqual(variant["alpha"], 16)
        self.assertEqual(len(variant["target_modules"]), 37)
        self.assertEqual(variant["expected_trainable_parameters"], 1_485_312)
        self.assertEqual(spec["fit"]["effective_batch_size"], 64)
        self.assertEqual(spec["selection"]["action_error_inference_noise_seeds"], [
            2026071835,
            2026071935,
            2026072035,
            2026072135,
        ])
        self.assertEqual(
            spec["closed_loop_opening"]["minimum_each_task_action_mse_reduction_fraction"],
            0.02,
        )
        self.assertFalse(spec["boundary"]["writer_authorized"])
        self.assertTrue(spec["boundary"]["no_validation_or_held_or_locked_access"])

    def test_contract_drift_fails_closed(self) -> None:
        changed = self.config.read_text(encoding="utf-8").replace(
            "drift_proxy_max = 0.02", "drift_proxy_max = 0.03"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaises(GateZeroActionAlignedContractError):
                load_action_aligned_acquisition_spec(
                    path,
                    gate_zero_path=self.gate_zero,
                    phase0_path=self.phase0,
                    competence_path=self.competence,
                )

    def test_canonical_fit_dispatch_accepts_the_action_aligned_contract(self) -> None:
        spec = load_oracle_fit_spec(
            self.config,
            gate_zero_path=self.gate_zero,
            phase0_path=self.phase0,
            competence_path=self.competence,
        )
        self.assertEqual(spec["variants"], ["action_aligned_official_default_r32"])

    def test_full_sampler_action_loss_backpropagates_through_generated_actions(self) -> None:
        owner = _ToyActionOwner()
        model = _ToyPeftWrapper(owner)
        target = torch.ones(2, 3, 2)
        batch = {"features": 2 * target, "action": target}
        noise = torch.randn(2, 3, 3)

        loss = full_sampler_generated_action_mse(model, batch, noise=noise)
        loss.backward()

        self.assertEqual(owner.reset_count, 1)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(owner.scale.grad)
        self.assertTrue(torch.isfinite(owner.scale.grad))

    def test_action_mse_selection_uses_primary_metric_and_drift_cap(self) -> None:
        candidates = [
            {"step": 0, "query_action_mse_mean": 1.0, "action_drift_proxy": 0.0},
            {"step": 5, "query_action_mse_mean": 0.70, "action_drift_proxy": 0.03},
            {"step": 10, "query_action_mse_mean": 0.80, "action_drift_proxy": 0.01},
        ]

        selected = select_action_mse_candidate(candidates, drift_proxy_max=0.02)

        self.assertEqual(selected["step"], 10)

    def test_canonical_launcher_dry_run_uses_the_single_fit_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_fit.sh"),
                f"--config={self.config}",
                "--variant=action_aligned_official_default_r32",
                "--task-id=3",
                "--gpu=4",
                "--output-dir=/tmp/ember-gate0-action-aligned-fit-test",
                "--latest-link=/tmp/ember-gate0-action-aligned-latest-test",
                "--stop-after-step=1",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_oracle_fit", completed.stdout)
        self.assertIn("--stop-after-step 1", completed.stdout)
        self.assertNotIn("gate_zero_query_action_alignment", completed.stdout)


if __name__ == "__main__":
    unittest.main()
