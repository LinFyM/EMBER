from __future__ import annotations

import json
import hashlib
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember import gate_zero_checkpoint as checkpoint_module  # noqa: E402
from ember.gate_zero_checkpoint import (  # noqa: E402
    CHECKPOINT_MANIFEST,
    GateZeroCheckpointError,
    build_policy_runtime_manifest,
    load_source_base_training_state_without_rng,
    restore_source_base_checkpoint_rng,
    rotate_source_base_recovery_checkpoints,
    save_source_base_checkpoint,
    validate_hashed_tree,
    validate_source_base_checkpoint,
)
from lerobot.utils.random_utils import serialize_rng_state  # noqa: E402


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


def _metadata(world_size: int = 1) -> dict:
    return {
        "checkpoint_role": "source_base_training_recovery",
        "topology": {
            "world_size": world_size,
            "micro_batch_size": 64 // world_size,
            "gradient_accumulation_steps": 1,
            "num_workers": 4 // world_size,
            "global_effective_batch_size": 64,
            "total_num_workers": 4,
            "global_slot_algorithm": "absolute_optimizer_step_accumulation_rank_local_slot_v1",
            "flow_input_authority": "rank0_global_native_sample_then_contiguous_scatter_v1",
            "checkpoint_writer_rank": 0,
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

    def test_distributed_checkpoint_saves_and_restores_every_rank_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for world_size in (2, 4):
                checkpoint = Path(temporary) / f"world_{world_size}" / "000001"
                rank_states = []
                expected_draws = []
                for rank in range(world_size):
                    random.seed(100 + rank)
                    np.random.seed(200 + rank)
                    torch.manual_seed(300 + rank)
                    rank_states.append(serialize_rng_state())
                    expected_draws.append(
                        (random.random(), float(np.random.random()), torch.rand(3))
                    )
                policy = _FakePolicy()
                optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                policy(torch.ones(1, 3)).sum().backward()
                optimizer.step()

                manifest = save_source_base_checkpoint(
                    checkpoint,
                    step=1,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=_FakeProcessor("policy_preprocessor"),
                    postprocessor=_FakeProcessor("policy_postprocessor"),
                    metadata=_metadata(world_size),
                    rank_rng_states=rank_states,
                )

                self.assertEqual(manifest["schema_version"], 3)
                self.assertEqual(manifest["distributed_rng"]["world_size"], world_size)
                self.assertEqual(
                    sorted((checkpoint / "distributed_rng").glob("rank_*.safetensors")),
                    [
                        checkpoint / "distributed_rng" / f"rank_{rank:05d}.safetensors"
                        for rank in range(world_size)
                    ],
                )
                validate_source_base_checkpoint(checkpoint, expected=_metadata(world_size))
                for rank, expected in enumerate(expected_draws):
                    restore_source_base_checkpoint_rng(
                        checkpoint, rank=rank, world_size=world_size
                    )
                    self.assertEqual(random.random(), expected[0])
                    self.assertEqual(float(np.random.random()), expected[1])
                    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
                with self.assertRaisesRegex(GateZeroCheckpointError, "topology"):
                    restore_source_base_checkpoint_rng(
                        checkpoint, rank=0, world_size=1
                    )

    def test_distributed_checkpoint_rejects_non_integer_manifest_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            save_source_base_checkpoint(
                checkpoint,
                step=1,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=_FakeProcessor("policy_preprocessor"),
                postprocessor=_FakeProcessor("policy_postprocessor"),
                metadata=_metadata(2),
                rank_rng_states=[serialize_rng_state(), serialize_rng_state()],
            )
            manifest_path = checkpoint / CHECKPOINT_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["topology"]["world_size"] = "2"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (checkpoint.parent / "000001.manifest.sha256").write_text(
                f"{digest}  {manifest_path.name}\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(GateZeroCheckpointError, "world size"):
                validate_source_base_checkpoint(checkpoint)

    def test_failed_checkpoint_never_publishes_partial_state(self) -> None:
        class _FailingProcessor(_FakeProcessor):
            def save_pretrained(self, directory: Path) -> None:
                raise RuntimeError("injected processor failure")

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            with self.assertRaisesRegex(RuntimeError, "injected"):
                save_source_base_checkpoint(
                    checkpoint,
                    step=1,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=_FailingProcessor("policy_preprocessor"),
                    postprocessor=_FakeProcessor("policy_postprocessor"),
                    metadata=_metadata(),
                )

            self.assertFalse(checkpoint.exists())
            self.assertFalse((checkpoint.parent / "000001.manifest.sha256").exists())
            self.assertFalse((checkpoint.parent / "last").exists())
            self.assertFalse(any(checkpoint.parent.glob(".*.tmp-*")))

    def test_post_publish_sidecar_failure_rolls_back_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "000001"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            with mock.patch(
                "ember.gate_zero_checkpoint._atomic_write_sidecar",
                side_effect=OSError("injected sidecar failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected sidecar"):
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

            self.assertFalse(checkpoint.exists())
            self.assertFalse((checkpoint.parent / "000001.manifest.sha256").exists())
            self.assertFalse((checkpoint.parent / "last").exists())
            self.assertFalse(any(checkpoint.parent.glob(".*.tmp-*")))

    def test_post_publish_last_failure_restores_previous_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            policy = _FakePolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            save_source_base_checkpoint(
                root / "000001",
                step=1,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=_FakeProcessor("policy_preprocessor"),
                postprocessor=_FakeProcessor("policy_postprocessor"),
                metadata=_metadata(),
            )
            real_fsync = checkpoint_module._fsync_directory
            fsync_calls = 0

            def fail_after_last_replace(path: Path) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 4:
                    raise OSError("injected last failure")
                real_fsync(path)

            with mock.patch(
                "ember.gate_zero_checkpoint._fsync_directory",
                side_effect=fail_after_last_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected last"):
                    save_source_base_checkpoint(
                        root / "000002",
                        step=2,
                        policy=policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        preprocessor=_FakeProcessor("policy_preprocessor"),
                        postprocessor=_FakeProcessor("policy_postprocessor"),
                        metadata={
                            **_metadata(),
                            "sampler": {
                                **_metadata()["sampler"],
                                "next_optimizer_step": 2,
                            },
                        },
                    )

            self.assertTrue((root / "000001").is_dir())
            self.assertFalse((root / "000002").exists())
            self.assertFalse((root / "000002.manifest.sha256").exists())
            self.assertEqual((root / "last").readlink(), Path("000001"))
            self.assertFalse(any(root.glob(".*.tmp-*")))

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
