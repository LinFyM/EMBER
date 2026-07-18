from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_checkpoint import (  # noqa: E402
    GateZeroCheckpointError,
    build_policy_runtime_manifest,
    load_source_base_training_state_without_rng,
    restore_source_base_checkpoint_rng,
    rotate_source_base_recovery_checkpoints,
    save_source_base_checkpoint,
    validate_hashed_tree,
    validate_source_base_checkpoint,
)


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    def save_pretrained(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps({"type": "smolvla", "use_peft": False}), encoding="utf-8"
        )
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
            directory / "model.safetensors",
        )


class _FakeProcessor:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.name}.json").write_text("{}", encoding="utf-8")


def _metadata() -> dict:
    return {
        "checkpoint_role": "source_base_training_recovery",
        "topology": {
            "world_size": 1,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "num_workers": 4,
        },
        "authorities": {
            "base_revision": "c" * 40,
            "base_weight_sha256": "d" * 64,
            "normalization_sha256": "e" * 64,
            "gate_zero_contract_sha256": "f" * 64,
            "phase0_contract_sha256": "a" * 64,
            "canonical_manifest_sha256": "b" * 64,
        },
        "sampler": {"seed": 17, "next_optimizer_step": 1},
    }


class GateZeroCheckpointTest(unittest.TestCase):
    def test_runtime_manifest_binds_role_authorities_and_every_policy_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            (policy_dir / "config.json").write_text(
                json.dumps({"type": "smolvla", "use_peft": False}), encoding="utf-8"
            )
            (policy_dir / "model.safetensors").write_bytes(b"model")
            (policy_dir / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
            (policy_dir / "policy_postprocessor.json").write_text("{}", encoding="utf-8")

            manifest = build_policy_runtime_manifest(
                policy_dir,
                policy_role="source_base_training_recovery",
                training_step=1000,
                base_revision="c" * 40,
                base_weight_sha256="d" * 64,
                normalization_sha256="e" * 64,
                contract_sha256="f" * 64,
            )

            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["artifact_kind"], "full_policy")
            self.assertEqual(manifest["policy_role"], "source_base_training_recovery")
            self.assertEqual(set(manifest["files"]), {
                "config.json",
                "model.safetensors",
                "policy_postprocessor.json",
                "policy_preprocessor.json",
            })
            validate_hashed_tree(policy_dir, manifest["files"])
            (policy_dir / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(GateZeroCheckpointError, "hash|bytes"):
                validate_hashed_tree(policy_dir, manifest["files"])

    def test_checkpoint_is_atomic_hash_bound_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            policy(torch.ones(1, 3)).sum().backward()
            optimizer.step()
            scheduler.step()

            manifest = save_source_base_checkpoint(
                checkpoint,
                step=1,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=_FakeProcessor("policy_preprocessor"),
                postprocessor=_FakeProcessor("policy_postprocessor"),
                metadata=_metadata(),
            )

            self.assertEqual(manifest["step"], 1)
            self.assertEqual((checkpoint.parent / "last").resolve(), checkpoint.resolve())
            self.assertTrue((checkpoint.parent / "000001.manifest.sha256").is_file())
            self.assertFalse(any(checkpoint.parent.glob(".*.tmp-*")))
            validated = validate_source_base_checkpoint(checkpoint, expected=_metadata())
            self.assertEqual(validated["files"], manifest["files"])
            with self.assertRaisesRegex(GateZeroCheckpointError, "overwrite"):
                save_source_base_checkpoint(
                    checkpoint,
                    step=1,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=_FakeProcessor("policy_preprocessor"),
                    postprocessor=_FakeProcessor("policy_postprocessor"),
                    metadata=_metadata(),
                )

    def test_resume_loads_state_before_explicit_rng_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            policy(torch.ones(1, 3)).sum().backward()
            optimizer.step()
            scheduler.step()
            saved_model = {name: value.detach().clone() for name, value in policy.state_dict().items()}
            random.seed(23)
            np.random.seed(23)
            torch.manual_seed(23)
            save_source_base_checkpoint(
                checkpoint,
                step=1,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=_FakeProcessor("policy_preprocessor"),
                postprocessor=_FakeProcessor("policy_postprocessor"),
                metadata=_metadata(),
            )
            expected_torch_draw = torch.rand(3)

            resumed_policy = _FakePolicy()
            resumed_optimizer = torch.optim.AdamW(resumed_policy.parameters(), lr=1e-4)
            resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
                resumed_optimizer, lambda _: 1.0
            )
            torch.manual_seed(999)
            rng_before_load = torch.get_rng_state().clone()
            step, resumed_optimizer, resumed_scheduler = load_source_base_training_state_without_rng(
                checkpoint,
                policy=resumed_policy,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                expected=_metadata(),
            )

            self.assertEqual(step, 1)
            torch.testing.assert_close(torch.get_rng_state(), rng_before_load, rtol=0, atol=0)
            for name, value in resumed_policy.state_dict().items():
                torch.testing.assert_close(value, saved_model[name], rtol=0, atol=0)
            restore_source_base_checkpoint_rng(checkpoint)
            torch.testing.assert_close(torch.rand(3), expected_torch_draw, rtol=0, atol=0)

    def test_checkpoint_validation_rejects_payload_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            policy(torch.ones(1, 3)).sum().backward()
            optimizer.step()
            save_source_base_checkpoint(
                checkpoint,
                step=1,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=_FakeProcessor("policy_preprocessor"),
                postprocessor=_FakeProcessor("policy_postprocessor"),
                metadata=_metadata(),
            )
            (checkpoint / "pretrained_model" / "model.safetensors").write_bytes(b"tampered")

            with self.assertRaisesRegex(GateZeroCheckpointError, "hash|bytes"):
                validate_source_base_checkpoint(checkpoint, expected=_metadata())

    def test_rotation_keeps_two_newest_recovery_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_root = Path(temporary) / "checkpoints"
            for step in (1, 2, 3):
                policy = _FakePolicy()
                optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                policy(torch.ones(1, 3)).sum().backward()
                optimizer.step()
                metadata = _metadata()
                metadata["sampler"] = {"seed": 17, "next_optimizer_step": step}
                save_source_base_checkpoint(
                    checkpoint_root / f"{step:06d}",
                    step=step,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=_FakeProcessor("policy_preprocessor"),
                    postprocessor=_FakeProcessor("policy_postprocessor"),
                    metadata=metadata,
                )

            removed = rotate_source_base_recovery_checkpoints(checkpoint_root, keep=2)

            self.assertEqual(removed, [1])
            self.assertFalse((checkpoint_root / "000001").exists())
            self.assertFalse((checkpoint_root / "000001.manifest.sha256").exists())
            self.assertTrue((checkpoint_root / "000002").is_dir())
            self.assertTrue((checkpoint_root / "000003").is_dir())
            self.assertEqual((checkpoint_root / "last").resolve(), (checkpoint_root / "000003").resolve())


if __name__ == "__main__":
    unittest.main()
