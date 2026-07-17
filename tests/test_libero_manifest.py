from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.libero_manifest import (  # noqa: E402
    SCENE_RE,
    _bddl_authority,
    _factor_coverage,
    ManifestError,
    audit_demonstration_file,
    compute_normalization,
    load_hub_surface,
    render_report,
    write_artifacts,
)


def _write_demo_file(path: Path, *, task_name: str, language: str) -> None:
    env_args = {
        "env_name": "Libero_Kitchen_Tabletop_Manipulation",
        "bddl_file": f"legacy/bddl_files/libero_100/{task_name}.bddl",
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": {
                "type": "OSC_POSE",
                "control_delta": True,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            },
            "camera_names": ["robot0_eye_in_hand", "agentview"],
            "camera_heights": 128,
            "camera_widths": 128,
            "use_camera_obs": True,
            "camera_depths": False,
            "control_freq": 20,
        },
    }
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = f"libero/libero/bddl_files/libero_90/{task_name}.bddl"
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["env_name"] = env_args["env_name"]
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["num_demos"] = 2
        data.attrs["problem_info"] = json.dumps(
            {
                "problem_name": "libero_kitchen_tabletop_manipulation",
                "domain_name": "robosuite",
                "language_instruction": language,
            }
        )
        data.attrs["tag"] = "libero-v1"
        data.attrs["total"] = 4
        for episode in range(2):
            demo = data.create_group(f"demo_{episode}")
            demo.attrs["init_state"] = np.arange(9, dtype=np.float64)
            demo.attrs["model_file"] = "/private/producer/path/secret.xml"
            demo.attrs["num_samples"] = 2
            actions = np.array(
                [[episode, 1, 2, 3, 4, 5, -1], [episode + 2, 3, 4, 5, 6, 7, 1]],
                dtype=np.float64,
            )
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("dones", data=np.array([0, 1], dtype=np.uint8))
            demo.create_dataset("rewards", data=np.array([0, 1], dtype=np.uint8))
            demo.create_dataset("robot_states", data=np.zeros((2, 9), dtype=np.float64))
            demo.create_dataset("states", data=np.zeros((2, 9), dtype=np.float64))
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
            obs.create_dataset("ee_ori", data=np.zeros((2, 3), dtype=np.float64))
            obs.create_dataset("ee_pos", data=np.zeros((2, 3), dtype=np.float64))
            obs.create_dataset(
                "ee_states",
                data=np.array([[episode, 1, 2, 3, 4, 5], [episode + 2, 3, 4, 5, 6, 7]], dtype=np.float64),
            )
            obs.create_dataset(
                "gripper_states",
                data=np.array([[episode, 1], [episode + 2, 3]], dtype=np.float64),
            )
            obs.create_dataset("joint_states", data=np.zeros((2, 7), dtype=np.float64))


class LiberoManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_hub_surface_filters_and_validates_lfs_authority(self) -> None:
        tree_path = self.root / "tree.json"
        tree_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "files": {
                        "README.md": {"blob_id": "a" * 40, "size": 10},
                        "libero_90/TASK_demo.hdf5": {
                            "blob_id": "b" * 40,
                            "size": 5,
                            "lfs_size": 5,
                            "lfs_sha256": "c" * 64,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        surface = load_hub_surface(
            tree_path,
            subdir="libero_90",
            expected_file_count=1,
            expected_total_bytes=5,
        )

        self.assertEqual(surface["file_count"], 1)
        self.assertEqual(surface["total_bytes"], 5)
        self.assertEqual(surface["files"]["TASK_demo.hdf5"]["sha256"], "c" * 64)
        with self.assertRaisesRegex(ManifestError, "total bytes"):
            load_hub_surface(
                tree_path,
                subdir="libero_90",
                expected_file_count=1,
                expected_total_bytes=6,
            )

    def test_scene_identity_supports_multiword_room_names(self) -> None:
        match = SCENE_RE.match("LIVING_ROOM_SCENE4_stack_the_bowls")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "LIVING_ROOM_SCENE4")

    def test_held_bddl_is_hashed_without_semantic_parser_access(self) -> None:
        bddl = self.root / "held.bddl"
        bddl.write_text("held semantics", encoding="utf-8")
        task = SimpleNamespace(bddl_file="held.bddl", language="allowed instruction")

        def forbidden_parser(_: str) -> None:
            raise AssertionError("held semantic parser must not run")

        record = _bddl_authority(
            task=task,
            split="held_out",
            bddl_path=bddl,
            parser=forbidden_parser,
            task_index=1,
        )

        self.assertEqual(
            set(record), {"filename", "sha256", "semantic_access_policy"}
        )
        self.assertEqual(record["semantic_access_policy"], "identity_only_not_parsed")

    def test_factor_coverage_marks_unread_held_semantics_as_not_evaluated(self) -> None:
        tasks = [
            {
                "scene": "KITCHEN_SCENE1",
                "split": "source",
                "bddl": {
                    "object_categories": ["source_object"],
                    "fixture_categories": ["source_fixture"],
                    "goal_state": [["in", "source_object", "source_fixture"]],
                },
            },
            {
                "scene": "KITCHEN_SCENE2",
                "split": "held_out",
                "bddl": {"semantic_access_policy": "identity_only_not_parsed"},
            },
        ]

        coverage = _factor_coverage(tasks)

        self.assertEqual(
            coverage["scenes"]["held_out_absent_from_source"],
            ["KITCHEN_SCENE2"],
        )
        self.assertEqual(
            coverage["scenes"]["held_out_coverage_status"],
            "evaluated_from_task_name_scene",
        )
        for dimension in ("object_categories", "fixture_categories", "goal_predicates"):
            self.assertIsNone(coverage[dimension]["held_out_absent_from_source"])
            self.assertEqual(
                coverage[dimension]["held_out_coverage_status"],
                "not_evaluated_due_to_access_policy",
            )

    def test_metadata_only_audit_never_returns_privileged_samples(self) -> None:
        task_name = "KITCHEN_SCENE1_test_task"
        path = self.root / f"{task_name}_demo.hdf5"
        _write_demo_file(path, task_name=task_name, language="test task")

        result = audit_demonstration_file(
            path,
            task_index=3,
            task_name=task_name,
            split="validation",
            language="test task",
            bddl_basename=f"{task_name}.bddl",
            expected_tag="libero-v1",
            expected_demos=2,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_bytes=path.stat().st_size,
            normalization_episodes=(),
        )

        self.assertIsNone(result.state_samples)
        self.assertIsNone(result.action_samples)
        serialized = json.dumps(result.record, sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("private/producer", serialized)
        self.assertNotIn("model_file", serialized)
        self.assertEqual(result.record["access_policy"], "metadata_only")
        self.assertEqual(result.record["controller"]["type"], "OSC_POSE")
        self.assertEqual(result.record["camera"]["names"], ["robot0_eye_in_hand", "agentview"])

    def test_legacy_env_bddl_basename_mismatch_is_a_provenance_note(self) -> None:
        task_name = "KITCHEN_SCENE1_test_task"
        path = self.root / f"{task_name}_demo.hdf5"
        _write_demo_file(path, task_name=task_name, language="test task")
        with h5py.File(path, "r+") as handle:
            env_args = json.loads(handle["data"].attrs["env_args"])
            env_args["bddl_file"] = "legacy/bddl_files/libero_100/old_task_wording.bddl"
            handle["data"].attrs.modify("env_args", json.dumps(env_args))

        result = audit_demonstration_file(
            path,
            task_index=0,
            task_name=task_name,
            split="source",
            language="test task",
            bddl_basename=f"{task_name}.bddl",
            expected_tag="libero-v1",
            expected_demos=2,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_bytes=path.stat().st_size,
            normalization_episodes=(),
        )

        warning_codes = {warning["code"] for warning in result.record["quality"]["warnings"]}
        self.assertIn("legacy_env_bddl_basename_mismatch", warning_codes)
        self.assertIn("legacy_env_bddl_suite", warning_codes)
        self.assertEqual(result.record["quality"]["status"], "pass_with_note")

    def test_canonical_hdf5_bddl_basename_mismatch_remains_fatal(self) -> None:
        task_name = "KITCHEN_SCENE1_test_task"
        path = self.root / f"{task_name}_demo.hdf5"
        _write_demo_file(path, task_name=task_name, language="test task")
        with h5py.File(path, "r+") as handle:
            handle["data"].attrs.modify(
                "bddl_file_name", "libero/libero/bddl_files/libero_90/wrong_task.bddl"
            )

        with self.assertRaisesRegex(ManifestError, "HDF5 BDDL basename mismatch"):
            audit_demonstration_file(
                path,
                task_index=0,
                task_name=task_name,
                split="source",
                language="test task",
                bddl_basename=f"{task_name}.bddl",
                expected_tag="libero-v1",
                expected_demos=2,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_bytes=path.stat().st_size,
                normalization_episodes=(),
            )

    def test_normalization_access_is_rejected_outside_source(self) -> None:
        task_name = "KITCHEN_SCENE1_test_task"
        path = self.root / f"{task_name}_demo.hdf5"
        _write_demo_file(path, task_name=task_name, language="test task")

        with self.assertRaisesRegex(ManifestError, "source tasks"):
            audit_demonstration_file(
                path,
                task_index=3,
                task_name=task_name,
                split="held_out",
                language="test task",
                bddl_basename=f"{task_name}.bddl",
                expected_tag="libero-v1",
                expected_demos=2,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_bytes=path.stat().st_size,
                normalization_episodes=(0,),
            )

    def test_source_normalization_contains_quantiles_and_provenance(self) -> None:
        task_name = "KITCHEN_SCENE1_test_task"
        path = self.root / f"{task_name}_demo.hdf5"
        _write_demo_file(path, task_name=task_name, language="test task")
        result = audit_demonstration_file(
            path,
            task_index=0,
            task_name=task_name,
            split="source",
            language="test task",
            bddl_basename=f"{task_name}.bddl",
            expected_tag="libero-v1",
            expected_demos=2,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_bytes=path.stat().st_size,
            normalization_episodes=(1,),
        )
        normalization = compute_normalization(
            [result], source_task_indices=[0], episode_bounds=[1, 1]
        )

        self.assertEqual(normalization["authority"]["split"], "source")
        self.assertEqual(normalization["authority"]["episode_pool"], "source_base_fit")
        self.assertEqual(normalization["observation.state"]["count"], 2)
        self.assertEqual(len(normalization["observation.state"]["q01"]), 8)
        self.assertEqual(len(normalization["action"]["q99"]), 7)
        self.assertEqual(result.record["access_policy"], "source_normalization_values")

    def test_html_report_has_filters_and_no_local_paths(self) -> None:
        manifest = {
            "summary": {"tasks": 1, "source": 1, "validation": 0, "held_out": 0},
            "tasks": [
                {
                    "task_index": 0,
                    "task_name": "KITCHEN_SCENE1_test_task",
                    "language": "test task",
                    "scene": "KITCHEN_SCENE1",
                    "split": "source",
                    "demonstrations": {"count": 50, "steps": 100},
                    "quality": {"status": "pass", "warning_count": 0},
                }
            ],
        }
        html = render_report(manifest, {"status": "pass", "issues": []})

        self.assertIn("split-filter", html)
        self.assertIn("KITCHEN_SCENE1_test_task", html)
        self.assertIn("manifest.json", html)
        self.assertNotIn(str(self.root), html)

    def test_artifacts_are_checksummed_and_latest_link_is_replaceable(self) -> None:
        manifest = {
            "summary": {
                "tasks": 1,
                "source": 1,
                "validation": 0,
                "held_out": 0,
                "demonstrations": 50,
                "frames": 100,
            },
            "tasks": [
                {
                    "task_index": 0,
                    "task_name": "KITCHEN_SCENE1_test_task",
                    "language": "test task",
                    "scene": "KITCHEN_SCENE1",
                    "split": "source",
                    "demonstrations": {"count": 50, "steps": 100},
                    "quality": {"status": "pass", "warning_count": 0},
                }
            ],
        }
        latest = self.root / "latest"
        first = self.root / "report-one"
        second = self.root / "report-two"
        kwargs = {
            "latest_link": latest,
            "manifest": manifest,
            "normalization": {"authority": {"split": "source"}},
            "quality_report": {"status": "pass", "issues": []},
        }

        write_artifacts(output_dir=first, **kwargs)
        write_artifacts(output_dir=second, **kwargs)

        self.assertTrue(latest.is_symlink())
        self.assertEqual(latest.resolve(), second.resolve())
        checksum_lines = (second / "checksums.sha256").read_text().splitlines()
        self.assertEqual(len(checksum_lines), 4)
        for line in checksum_lines:
            expected, filename = line.split("  ", maxsplit=1)
            self.assertEqual(
                hashlib.sha256((second / filename).read_bytes()).hexdigest(), expected
            )


if __name__ == "__main__":
    unittest.main()
