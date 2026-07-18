from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_checkpoint import (  # noqa: E402
    GateZeroCheckpointError,
    build_policy_runtime_manifest,
    validate_hashed_tree,
)


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


if __name__ == "__main__":
    unittest.main()
