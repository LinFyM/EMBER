from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_data import (  # noqa: E402
    GateZeroSurface,
    GateZeroDataError,
    Hdf5TaskAuthority,
    SourceHdf5Dataset,
    TaskDemoFrameBatchSampler,
    build_frame_index,
    load_surface_authorities,
)


class GateZeroDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.path = self.root / "task_demo.hdf5"
        with h5py.File(self.path, "w") as handle:
            data = handle.create_group("data")
            for demo_index, length in enumerate((3, 2)):
                demo = data.create_group(f"demo_{demo_index}")
                demo.attrs["num_samples"] = length
                obs = demo.create_group("obs")
                base = np.arange(length * 128 * 128 * 3, dtype=np.uint32)
                base = np.asarray(base % 256, dtype=np.uint8).reshape(length, 128, 128, 3)
                obs.create_dataset("agentview_rgb", data=base)
                obs.create_dataset("eye_in_hand_rgb", data=base + 1)
                obs.create_dataset(
                    "ee_states", data=np.arange(length * 6, dtype=np.float64).reshape(length, 6)
                )
                obs.create_dataset(
                    "gripper_states", data=np.arange(length * 2, dtype=np.float64).reshape(length, 2)
                )
                demo.create_dataset(
                    "actions", data=np.arange(length * 7, dtype=np.float64).reshape(length, 7)
                )
        self.authority = Hdf5TaskAuthority(
            task_id=3,
            language="do the task",
            path=self.path,
            expected_bytes=self.path.stat().st_size,
            expected_sha256=None,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_index_is_deterministic_and_scoped_to_declared_demos(self) -> None:
        index = build_frame_index([self.authority], demo_indices=[1], verify_sha256=False)

        self.assertEqual(index, [(3, 1, 0), (3, 1, 1)])

    def test_sample_flips_each_camera_once_and_builds_eight_dimensional_state(self) -> None:
        dataset = SourceHdf5Dataset(
            [self.authority], demo_indices=[0], action_chunk_size=2, verify_sha256=False
        )

        sample = dataset[0]
        with h5py.File(self.path, "r") as handle:
            raw = np.asarray(handle["data/demo_0/obs/agentview_rgb"][0])
        np.testing.assert_array_equal(
            sample["observation.images.camera1"], raw[::-1, ::-1].transpose(2, 0, 1)
        )
        self.assertEqual(sample["observation.images.camera1"].shape, (3, 128, 128))
        self.assertEqual(sample["observation.state"].shape, (8,))
        self.assertEqual(sample["action"].shape, (2, 7))
        np.testing.assert_array_equal(sample["action_is_pad"], [False, False])
        self.assertEqual(sample["task"], "do the task")

    def test_terminal_chunk_repeats_last_action_and_is_masked(self) -> None:
        dataset = SourceHdf5Dataset(
            [self.authority], demo_indices=[1], action_chunk_size=3, verify_sha256=False
        )

        sample = dataset[1]
        np.testing.assert_array_equal(sample["action_is_pad"], [False, True, True])
        np.testing.assert_array_equal(
            sample["action"][1:], np.repeat(sample["action"][:1], 2, axis=0)
        )

    def test_unknown_demo_fails_closed(self) -> None:
        with self.assertRaisesRegex(GateZeroDataError, "demo_2"):
            build_frame_index([self.authority], demo_indices=[2], verify_sha256=False)

    def test_hierarchical_sampler_is_step_resumable_and_not_frame_weighted(self) -> None:
        dataset = SourceHdf5Dataset(
            [self.authority], demo_indices=[0, 1], action_chunk_size=2, verify_sha256=False
        )
        complete = list(
            TaskDemoFrameBatchSampler(
                dataset,
                micro_batch_size=6,
                optimizer_steps=3,
                gradient_accumulation_steps=2,
                seed=17,
                start_optimizer_step=0,
            )
        )
        resumed = list(
            TaskDemoFrameBatchSampler(
                dataset,
                micro_batch_size=6,
                optimizer_steps=1,
                gradient_accumulation_steps=2,
                seed=17,
                start_optimizer_step=2,
            )
        )

        self.assertEqual(complete[4:], resumed)
        chosen_demos = [dataset.frame_index[index][1] for batch in complete for index in batch]
        self.assertIn(0, chosen_demos)
        self.assertIn(1, chosen_demos)

        long_run = TaskDemoFrameBatchSampler(
            dataset,
            micro_batch_size=100,
            optimizer_steps=100,
            gradient_accumulation_steps=1,
            seed=19,
        )
        counts = {0: 0, 1: 0}
        for batch in long_run:
            for index in batch:
                counts[dataset.frame_index[index][1]] += 1
        self.assertLess(abs(counts[0] / sum(counts.values()) - 0.5), 0.03)

    def test_effective_batch_draws_are_invariant_to_microbatch_partition(self) -> None:
        dataset = SourceHdf5Dataset(
            [self.authority], demo_indices=[0, 1], action_chunk_size=2, verify_sha256=False
        )
        batches_8 = list(
            TaskDemoFrameBatchSampler(
                dataset,
                micro_batch_size=8,
                optimizer_steps=1,
                gradient_accumulation_steps=8,
                seed=23,
            )
        )
        batches_16 = list(
            TaskDemoFrameBatchSampler(
                dataset,
                micro_batch_size=16,
                optimizer_steps=1,
                gradient_accumulation_steps=4,
                seed=23,
            )
        )

        self.assertEqual(
            [index for batch in batches_8 for index in batch],
            [index for batch in batches_16 for index in batch],
        )

    def test_distributed_rank_shards_reconstruct_single_rank_draws_without_overlap(self) -> None:
        dataset = SourceHdf5Dataset(
            [self.authority], demo_indices=[0, 1], action_chunk_size=2, verify_sha256=False
        )
        single = list(
            TaskDemoFrameBatchSampler(
                dataset,
                micro_batch_size=64,
                optimizer_steps=2,
                gradient_accumulation_steps=1,
                seed=29,
                rank=0,
                world_size=1,
                global_effective_batch_size=64,
            )
        )
        for world_size, micro_batch_size in ((2, 32), (4, 16)):
            rank_batches = [
                list(
                    TaskDemoFrameBatchSampler(
                        dataset,
                        micro_batch_size=micro_batch_size,
                        optimizer_steps=2,
                        gradient_accumulation_steps=1,
                        seed=29,
                        rank=rank,
                        world_size=world_size,
                        global_effective_batch_size=64,
                    )
                )
                for rank in range(world_size)
            ]
            reconstructed = [
                [index for rank in range(world_size) for index in rank_batches[rank][step]]
                for step in range(2)
            ]
            self.assertEqual(reconstructed, single)

    def test_surface_factory_denies_report_without_selection_freeze(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task_index": 3,
                            "task_name": "task",
                            "language": "do the task",
                            "split": "source",
                            "hdf5": {
                                "filename": self.path.name,
                                "bytes": self.path.stat().st_size,
                                "sha256": "a" * 64,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        spec = {
            "name": "pilot",
            "authority": {"canonical_manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()},
            "access": {
                "source_base_fit": [0, 0],
                "oracle_support": [0, 0],
                "functional_query": [0, 0],
                "locked_source_report": [0, 0],
            },
            "data": {"task_ids": [3]},
        }
        phase0 = {"splits": {"source": [3], "validation": [], "held_out": []}}

        with self.assertRaisesRegex(GateZeroDataError, "selection-freeze"):
            load_surface_authorities(
                spec,
                phase0,
                manifest_path=manifest,
                dataset_root=self.root,
                surface=GateZeroSurface.REPORT,
            )

    def test_support_surface_requires_one_declared_oracle_task(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task_index": 3,
                            "language": "do the task",
                            "split": "source",
                            "hdf5": {
                                "filename": self.path.name,
                                "bytes": self.path.stat().st_size,
                                "sha256": "a" * 64,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        spec = {
            "name": "pilot",
            "authority": {"canonical_manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()},
            "access": {"oracle_support": [0, 0]},
            "data": {"task_ids": [3]},
        }
        phase0 = {"splits": {"source": [3], "validation": [], "held_out": []}}

        with self.assertRaisesRegex(GateZeroDataError, "oracle_task_id"):
            load_surface_authorities(
                spec,
                phase0,
                manifest_path=manifest,
                dataset_root=self.root,
                surface=GateZeroSurface.SUPPORT,
            )


if __name__ == "__main__":
    unittest.main()
