from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.runtime_env import RuntimeEnvironmentError, repair_runtime_environment  # noqa: E402


def _metadata(name: str, version: str) -> str:
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"


class RuntimeEnvironmentRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.site_packages = Path(self.temporary_directory.name)

        bddl_metadata = self.site_packages / "bddl-1.0.1.dist-info" / "METADATA"
        bddl_metadata.parent.mkdir()
        bddl_metadata.write_text(_metadata("bddl", "1.0.1"), encoding="utf-8")
        (self.site_packages / "bddl-1.0.1.egg-info").write_text(
            bddl_metadata.read_text(encoding="utf-8"), encoding="utf-8"
        )

        lerobot_metadata = (
            self.site_packages / "lerobot-0.6.0.dist-info" / "METADATA"
        )
        lerobot_metadata.parent.mkdir()
        lerobot_metadata.write_text(
            _metadata("lerobot", "0.6.0"), encoding="utf-8"
        )
        (self.site_packages / "lerobot-0.6.0.egg-info").write_text(
            lerobot_metadata.read_text(encoding="utf-8"), encoding="utf-8"
        )

        robosuite_metadata = (
            self.site_packages / "robosuite-1.4.0.dist-info" / "METADATA"
        )
        robosuite_metadata.parent.mkdir()
        robosuite_metadata.write_text(
            _metadata("robosuite", "1.4.0"), encoding="utf-8"
        )
        robosuite_package = self.site_packages / "robosuite"
        robosuite_package.mkdir()
        (robosuite_package / "macros.py").write_text(
            'FILE_LOGGING_LEVEL = "DEBUG"\n', encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_repair_is_scoped_and_idempotent(self) -> None:
        first = repair_runtime_environment(self.site_packages)
        second = repair_runtime_environment(self.site_packages)

        self.assertEqual(
            first,
            {
                "bddl_metadata_removed": True,
                "lerobot_metadata_removed": True,
                "robosuite_override_created": True,
            },
        )
        self.assertEqual(
            second,
            {
                "bddl_metadata_removed": False,
                "lerobot_metadata_removed": False,
                "robosuite_override_created": False,
            },
        )
        self.assertFalse((self.site_packages / "bddl-1.0.1.egg-info").exists())
        self.assertFalse((self.site_packages / "lerobot-0.6.0.egg-info").exists())
        override = self.site_packages / "robosuite" / "macros_private.py"
        self.assertIn("_macros.FILE_LOGGING_LEVEL = None", override.read_text())

    def test_repair_refuses_unrecognized_bddl_payload(self) -> None:
        (self.site_packages / "bddl-1.0.1.egg-info").write_text(
            "unexpected metadata", encoding="utf-8"
        )

        with self.assertRaisesRegex(RuntimeEnvironmentError, "does not match"):
            repair_runtime_environment(self.site_packages)

    def test_repair_refuses_to_overwrite_private_macros(self) -> None:
        override = self.site_packages / "robosuite" / "macros_private.py"
        override.write_text("# user-owned settings\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeEnvironmentError, "Refusing to overwrite"):
            repair_runtime_environment(self.site_packages)

    def test_locked_pyav_backend_decodes_timestamped_frames(self) -> None:
        import av
        import numpy as np
        import torch
        from lerobot.datasets.video_utils import decode_video_frames

        video_path = self.site_packages / "decoder-smoke.mp4"
        with av.open(video_path, mode="w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            for value in (0, 80, 160):
                pixels = np.full((16, 16, 3), value, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        frames = decode_video_frames(
            video_path,
            timestamps=[0.0, 0.1, 0.2],
            tolerance_s=0.051,
            backend="pyav",
            return_uint8=True,
        )

        self.assertEqual(tuple(frames.shape), (3, 3, 16, 16))
        self.assertEqual(frames.dtype, torch.uint8)
        self.assertLess(frames[0].float().mean(), frames[-1].float().mean())


if __name__ == "__main__":
    unittest.main()
