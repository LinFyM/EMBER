from __future__ import annotations

import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_oracle_contract import (  # noqa: E402
    GateZeroOracleContractError,
    load_oracle_execution_spec,
)
from ember.gate_zero_oracle_fit import (  # noqa: E402
    build_oracle_optimizer,
    capture_trainable_state,
    evenly_spaced_anchor_indices,
    select_drift_safe_candidate,
    validate_fit_job,
)
from ember.gate_zero_oracle_artifacts import (  # noqa: E402
    GateZeroOracleArtifactError,
    load_recovery_artifact,
    publish_selected_artifact,
    save_candidate_artifact,
    save_recovery_artifact,
    validate_candidate_artifact,
    validate_selected_artifact,
)
from ember.gate_zero_oracle_metrics import (  # noqa: E402
    anchor_flat_indices,
    summarize_action_chunk_errors,
    unit_variance_mean_action_kl,
)


class GateZeroOracleExecutionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.execution_path = ROOT / "configs" / "gate_zero_oracle_execution.toml"
        self.gate_zero_path = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        self.phase0_path = ROOT / "configs" / "phase0.toml"
        self.competence_path = ROOT / "configs" / "gate_zero_source_competence.toml"

    def load(self):
        return load_oracle_execution_spec(
            self.execution_path,
            gate_zero_path=self.gate_zero_path,
            phase0_path=self.phase0_path,
            competence_path=self.competence_path,
        )

    def test_checked_in_execution_contract_preserves_primary_and_capacity_roles(self) -> None:
        spec = self.load()

        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["variants"], ["lora", "partial_upper_bound"])
        self.assertEqual(spec["fit"]["candidate_steps"], [0, 250, 500, 1000, 2000])
        self.assertEqual(spec["fit"]["lora"]["expected_trainable_parameters"], 40320)
        self.assertFalse(spec["fit"]["partial_upper_bound"]["matched_baseline"])
        self.assertFalse(spec["fit"]["partial_upper_bound"]["may_authorize_writer"])
        self.assertFalse(spec["selection"]["selection_uses_locked_report"])
        self.assertEqual(spec["report"]["official_rollout_init_state_indices"], list(range(16, 24)))
        self.assertFalse(spec["decision"]["writer_authorized_by_this_two_task_pilot"])

    def test_parent_contract_hash_drift_fails_closed(self) -> None:
        changed = self.execution_path.read_text(encoding="utf-8").replace(
            "7bbda2723c72c60c6cfc39c10acaf23a05f4ea0af724c054792cd00c890cfd98",
            "0" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(GateZeroOracleContractError, "Gate 0 contract SHA256"):
                load_oracle_execution_spec(
                    path,
                    gate_zero_path=self.gate_zero_path,
                    phase0_path=self.phase0_path,
                    competence_path=self.competence_path,
                )

    def test_anchor_schedule_is_inclusive_deterministic_and_unique(self) -> None:
        self.assertEqual(evenly_spaced_anchor_indices(15, 8), [0, 2, 4, 6, 8, 10, 12, 14])
        self.assertEqual(evenly_spaced_anchor_indices(8, 8), list(range(8)))
        with self.assertRaisesRegex(ValueError, "fewer frames"):
            evenly_spaced_anchor_indices(7, 8)

    def test_selection_uses_query_only_and_rejects_lower_loss_unsafe_drift(self) -> None:
        candidates = [
            {"step": 0, "query_flow_mse": 1.0, "action_drift_proxy": 0.0},
            {"step": 250, "query_flow_mse": 0.70, "action_drift_proxy": 0.03},
            {"step": 500, "query_flow_mse": 0.80, "action_drift_proxy": 0.01},
            {"step": 1000, "query_flow_mse": 0.85, "action_drift_proxy": 0.01},
        ]

        selected = select_drift_safe_candidate(candidates, drift_proxy_max=0.02)

        self.assertEqual(selected["step"], 500)
        self.assertEqual(selected["query_flow_mse"], 0.80)

    def test_unit_variance_gaussian_mean_action_kl_uses_normalized_action_scalars(self) -> None:
        base = torch.zeros(2, 3, 2)
        adapted = torch.ones(2, 3, 2)

        self.assertEqual(unit_variance_mean_action_kl(adapted, base), 0.5)
        with self.assertRaisesRegex(ValueError, "shape"):
            unit_variance_mean_action_kl(adapted[:, :2], base)

    def test_anchor_flat_indices_group_by_demo_and_use_frozen_schedule(self) -> None:
        frame_index = tuple(
            [(3, 40, frame) for frame in range(8)]
            + [(3, 41, frame) for frame in range(15)]
        )

        selected = anchor_flat_indices(frame_index, anchor_count=8)

        self.assertEqual(selected[:8], list(range(8)))
        self.assertEqual(selected[8:], [8, 10, 12, 14, 16, 18, 20, 22])

    def test_action_chunk_error_summary_keeps_row_episode_dimension_and_time_identity(self) -> None:
        target = torch.zeros(2, 4, 2)
        predicted = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
                [[0.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.0]],
            ]
        )

        summary = summarize_action_chunk_errors(
            predicted,
            target,
            ["task3/demo40/frame0", "task3/demo41/frame7"],
            time_partition_count=2,
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["action_chunk_size"], 4)
        self.assertEqual(summary["action_dimension"], 2)
        self.assertEqual(summary["time_partitions"], [[0, 2], [2, 4]])
        self.assertEqual(summary["mean_squared_error"], 1.25)
        self.assertEqual(summary["by_row_mse"], {
            "task3/demo40/frame0": 1.25,
            "task3/demo41/frame7": 1.25,
        })
        self.assertEqual(summary["by_episode_mse"], {"40": 1.25, "41": 1.25})
        self.assertEqual(summary["by_action_dimension_mse"], [1.25, 1.25])
        self.assertEqual(summary["by_time_partition_mse"], [0.5, 2.0])

    def test_action_chunk_error_summary_rejects_changed_identity_or_shape(self) -> None:
        value = torch.zeros(2, 4, 2)
        with self.assertRaisesRegex(ValueError, "row keys"):
            summarize_action_chunk_errors(value, value, ["task3/demo40/frame0"])
        with self.assertRaisesRegex(ValueError, "shape"):
            summarize_action_chunk_errors(value[:, :3], value, ["a", "b"])

    def test_candidate_artifact_is_atomic_hash_bound_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {
                "layer.a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "layer.b": torch.ones(2, dtype=torch.bfloat16),
            }
            metrics = {
                "step": 250,
                "query_flow_mse": 0.8,
                "action_drift_proxy": 0.01,
            }

            candidate = save_candidate_artifact(
                root,
                variant="lora",
                task_id=3,
                step=250,
                trainable_state=state,
                metrics=metrics,
                authorities={"execution_contract_sha256": "a" * 64},
            )
            manifest = validate_candidate_artifact(candidate)

            self.assertEqual(manifest["variant"], "lora")
            self.assertEqual(manifest["task_id"], 3)
            self.assertEqual(manifest["step"], 250)
            self.assertEqual(manifest["metrics"], metrics)
            selected = publish_selected_artifact(root, candidate)
            selected_manifest = validate_selected_artifact(selected)
            self.assertEqual(selected_manifest["selected_step"], 250)
            self.assertEqual(selected_manifest["selected_metrics"], metrics)
            with self.assertRaisesRegex(GateZeroOracleArtifactError, "overwrite"):
                save_candidate_artifact(
                    root,
                    variant="lora",
                    task_id=3,
                    step=250,
                    trainable_state=state,
                    metrics=metrics,
                    authorities={"execution_contract_sha256": "a" * 64},
                )
            (candidate / "trainable_state.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(GateZeroOracleArtifactError, "hash"):
                validate_candidate_artifact(candidate)

    def test_recovery_restores_trainable_optimizer_and_step_from_atomic_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            loss = model(torch.ones(1, 3)).square().sum()
            loss.backward()
            optimizer.step()
            expected = {name: value.detach().clone() for name, value in model.named_parameters()}

            recovery = save_recovery_artifact(
                root,
                variant="partial_upper_bound",
                task_id=4,
                step=250,
                trainable_state=expected,
                optimizer=optimizer,
                authorities={"execution_contract_sha256": "b" * 64},
            )
            with torch.no_grad():
                for value in model.parameters():
                    value.zero_()
            fresh_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

            restored_step = load_recovery_artifact(
                recovery,
                model=model,
                optimizer=fresh_optimizer,
                expected={"variant": "partial_upper_bound", "task_id": 4},
            )

            self.assertEqual(restored_step, 250)
            for name, value in model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]))
            self.assertTrue(fresh_optimizer.state)
            self.assertEqual((root / "recovery" / "last").resolve(), recovery.resolve())

    def test_fit_job_and_optimizer_preserve_predeclared_variant_contract(self) -> None:
        spec = self.load()
        variant_spec = validate_fit_job(spec, variant="lora", task_id=3)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        model[1].weight.requires_grad_(False)
        model[1].bias.requires_grad_(False)

        state = capture_trainable_state(model)
        optimizer = build_oracle_optimizer(model, variant_spec)

        self.assertEqual(set(state), {"0.bias", "0.weight"})
        self.assertEqual(optimizer.param_groups[0]["lr"], spec["fit"]["lora"]["learning_rate"])
        with self.assertRaisesRegex(Exception, "variant"):
            validate_fit_job(spec, variant="unknown", task_id=3)
        with self.assertRaisesRegex(Exception, "task"):
            validate_fit_job(spec, variant="lora", task_id=2)

    def test_single_launcher_dry_run_exposes_one_canonical_fit_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_gate_zero_oracle_fit.sh"),
                "--variant=lora",
                "--task-id=3",
                "--gpu=4",
                "--output-dir=/tmp/ember-gate0-fit-test",
                "--latest-link=/tmp/ember-gate0-fit-latest-test",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("-m ember.gate_zero_oracle_fit", completed.stdout)
        self.assertIn("--variant lora", completed.stdout)
        self.assertIn("--task-id 3", completed.stdout)
        self.assertNotIn("torch.distributed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
