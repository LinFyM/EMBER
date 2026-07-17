from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.eval_artifacts import (  # noqa: E402
    EvalArtifactError,
    build_eval_gallery,
    update_latest_link,
)


class EvalArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_dir = self.root / "run_001"
        video = self.run_dir / "videos" / "libero_spatial_0" / "episode_0.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fake-mp4-for-manifest")
        (self.run_dir / "eval_info.json").write_text(
            json.dumps(
                {
                    "per_task": [
                        {
                            "task_group": "libero_spatial",
                            "task_id": 0,
                            "metrics": {
                                "successes": [True],
                                "sum_rewards": [1.0],
                                "video_paths": [str(video)],
                            },
                        }
                    ],
                    "overall": {"pc_success": 100.0, "n_episodes": 1},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_gallery_is_self_contained_and_manifested(self) -> None:
        result = build_eval_gallery(self.run_dir)

        index = (self.run_dir / "index.html").read_text(encoding="utf-8")
        manifest = json.loads(
            (self.run_dir / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("libero_spatial / task 0", index)
        self.assertIn("<video controls", index)
        self.assertIn("videos/libero_spatial_0/episode_0.mp4", index)
        self.assertEqual(manifest["videos"][0]["bytes"], 21)
        self.assertEqual(len(manifest["videos"][0]["sha256"]), 64)
        self.assertEqual(result["video_count"], 1)

    def test_latest_link_is_atomic_and_refuses_regular_file(self) -> None:
        latest = self.root / "latest"
        update_latest_link(self.run_dir, latest)
        self.assertTrue(latest.is_symlink())
        self.assertEqual(latest.resolve(), self.run_dir.resolve())

        latest.unlink()
        latest.write_text("do not replace", encoding="utf-8")
        with self.assertRaisesRegex(EvalArtifactError, "non-symlink"):
            update_latest_link(self.run_dir, latest)

    def test_gallery_refuses_video_outside_run_root(self) -> None:
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        info_path = self.run_dir / "eval_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["per_task"][0]["metrics"]["video_paths"] = [str(outside)]
        info_path.write_text(json.dumps(info), encoding="utf-8")

        with self.assertRaisesRegex(EvalArtifactError, "outside run directory"):
            build_eval_gallery(self.run_dir)


if __name__ == "__main__":
    unittest.main()
