from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.phase0_runtime import (  # noqa: E402
    Phase0RuntimeError,
    WeightSnapshotSpec,
    materialize_policy_runtime_view,
    write_libero_config,
)


class Phase0RuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_policy_view_pins_constructor_and_tokenizer(self) -> None:
        source = self.root / "source"
        vlm = self.root / "vlm"
        output = self.root / "runtime"
        source.mkdir()
        vlm.mkdir()
        weight = source / "model.safetensors"
        vlm_weight = vlm / "model.safetensors"
        weight.write_bytes(b"policy-weight")
        vlm_weight.write_bytes(b"vlm-weight")
        (vlm / "config.json").write_text("{}\n", encoding="utf-8")
        (vlm / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (source / "config.json").write_text(
            json.dumps({"vlm_model_name": "upstream/vlm"}), encoding="utf-8"
        )
        (source / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "tokenizer_processor",
                            "config": {"tokenizer_name": "upstream/vlm"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        kwargs = {
            "source_policy": source,
            "vlm_snapshot": vlm,
            "output_policy": output,
            "upstream_vlm_name": "upstream/vlm",
            "source_spec": WeightSnapshotSpec(
                revision="1" * 40,
                weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                weight_bytes=weight.stat().st_size,
            ),
            "vlm_spec": WeightSnapshotSpec(
                revision="2" * 40,
                weight_sha256=hashlib.sha256(vlm_weight.read_bytes()).hexdigest(),
                weight_bytes=vlm_weight.stat().st_size,
            ),
        }
        first = materialize_policy_runtime_view(**kwargs)
        second = materialize_policy_runtime_view(**kwargs)

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertTrue((output / "model.safetensors").is_symlink())
        policy_config = json.loads((output / "config.json").read_text())
        processor_config = json.loads(
            (output / "policy_preprocessor.json").read_text()
        )
        expected_vlm = str(vlm.resolve())
        self.assertEqual(policy_config["vlm_model_name"], expected_vlm)
        self.assertEqual(
            processor_config["steps"][0]["config"]["tokenizer_name"],
            expected_vlm,
        )
        self.assertEqual(
            json.loads((output / "runtime_manifest.json").read_text())[
                "source_revision"
            ],
            "1" * 40,
        )

    def test_policy_view_refuses_wrong_weight(self) -> None:
        source = self.root / "source"
        vlm = self.root / "vlm"
        source.mkdir()
        vlm.mkdir()
        (source / "model.safetensors").write_bytes(b"wrong")
        (vlm / "model.safetensors").write_bytes(b"vlm")

        with self.assertRaisesRegex(Phase0RuntimeError, "source policy weight"):
            materialize_policy_runtime_view(
                source_policy=source,
                vlm_snapshot=vlm,
                output_policy=self.root / "runtime",
                upstream_vlm_name="upstream/vlm",
                source_spec=WeightSnapshotSpec("1" * 40, "0" * 64, 5),
                vlm_spec=WeightSnapshotSpec(
                    "2" * 40, hashlib.sha256(b"vlm").hexdigest(), 3
                ),
            )

    def test_libero_config_is_explicit_and_idempotent(self) -> None:
        site_packages = self.root / "site-packages"
        benchmark = site_packages / "libero" / "libero"
        (benchmark / "bddl_files").mkdir(parents=True)
        (benchmark / "init_files").mkdir()
        assets = self.root / "assets"
        for directory in (
            "articulated_objects",
            "stable_scanned_objects",
            "turbosquid_objects",
            "stable_hope_objects",
            "scenes",
        ):
            (assets / directory).mkdir(parents=True)
        data = self.root / "data"
        data.mkdir()
        config_root = self.root / "libero-config"

        first = write_libero_config(
            config_root=config_root,
            site_packages=site_packages,
            asset_snapshot=assets,
            data_root=data,
            expected_asset_file_count=0,
            expected_asset_bytes=0,
        )
        second = write_libero_config(
            config_root=config_root,
            site_packages=site_packages,
            asset_snapshot=assets,
            data_root=data,
            expected_asset_file_count=0,
            expected_asset_bytes=0,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        config = json.loads((config_root / "config.yaml").read_text())
        self.assertEqual(config["assets"], str(assets.resolve()))
        self.assertEqual(config["datasets"], str(data.resolve()))
        package_assets = benchmark / "assets"
        self.assertTrue(package_assets.is_symlink())
        self.assertEqual(package_assets.resolve(), assets.resolve())


if __name__ == "__main__":
    unittest.main()
