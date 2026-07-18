from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.video_information_probe import (  # noqa: E402
    VideoInformationProbeError,
    decide_video_probe,
    derive_clip_condition,
    fit_frozen_linear_probe,
    load_video_recovery_spec,
    load_video_spec,
    main,
    score_linear_probe,
    stratified_accuracy_interval,
    temporal_moment_descriptor,
    uniform_frame_indices,
)
from ember.video_probe_runtime import feature_descriptors  # noqa: E402


class VideoInformationProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "gate_minus1_video_information_probe.toml"
        self.recovery_path = ROOT / "configs" / "gate_minus1_video_information_recovery1.toml"

    def test_checked_in_protocol_is_source_only_and_frozen_before_outcomes(self) -> None:
        spec = load_video_spec(self.config_path)

        self.assertEqual(spec["task_ids"], [3, 4])
        self.assertEqual(spec["source_task_ids"], [3, 4])
        self.assertEqual(spec["support_demo_indices"], list(range(24)))
        self.assertEqual(spec["query_demo_indices"], list(range(24, 48)))
        self.assertEqual(spec["reserved_demo_indices"], [48, 49])
        self.assertFalse(spec["encoder"]["task_language_visible"])
        self.assertFalse(spec["encoder"]["task_id_visible"])
        self.assertFalse(spec["claim_boundary"]["gate_decision_authorized"])
        self.assertEqual(spec["resources"]["gpu_count"], 1)
        self.assertEqual(spec["resources"]["batch_size"], 48)

    def test_recovery_changes_only_the_frozen_representation_and_output_budget(self) -> None:
        base = load_video_spec(self.config_path)
        recovery = load_video_recovery_spec(self.recovery_path, self.config_path)

        self.assertEqual(recovery["schema_version"], 2)
        self.assertEqual(
            recovery["encoder"]["feature"],
            "smolvlm_visual_connector_temporal_moments_v1",
        )
        self.assertEqual(recovery["task_ids"], base["task_ids"])
        self.assertEqual(recovery["support_demo_indices"], base["support_demo_indices"])
        self.assertEqual(recovery["query_demo_indices"], base["query_demo_indices"])
        self.assertEqual(recovery["conditions"], base["conditions"])
        self.assertEqual(recovery["thresholds"], base["thresholds"])
        self.assertEqual(recovery["readout"], base["readout"])
        self.assertEqual(recovery["resources"]["batch_size"], 48)
        self.assertEqual(recovery["resources"]["expected_output_gib"], 0.03)
        self.assertFalse(recovery["recovery"]["representation"]["language_prompt_used"])

    def test_cli_preserves_existing_latest_symlink_as_a_lexical_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / "previous"
            previous.mkdir()
            latest = root / "latest"
            latest.symlink_to(previous.name, target_is_directory=True)
            output = root / "output"
            arguments = [
                "video_information_probe",
                "--config",
                str(root / "config.toml"),
                "--source-pair-config",
                str(root / "source.toml"),
                "--contract",
                str(root / "contract.toml"),
                "--seal",
                str(root / "seal.json"),
                "--manifest",
                str(root / "manifest.json"),
                "--dataset-root",
                str(root / "dataset"),
                "--model-path",
                str(root / "model"),
                "--output-dir",
                str(output),
                "--latest-link",
                str(latest),
                "--physical-gpu",
                "4",
            ]
            result = {
                "output_dir": str(output),
                "latest_link": str(latest),
                "status": "test",
                "artifact_bytes": 0,
                "video_count": 0,
                "wall_seconds": 0.0,
            }

            with patch.object(sys, "argv", arguments), patch(
                "ember.video_information_probe.run_probe", return_value=result
            ) as run:
                self.assertEqual(main(), 0)

            self.assertEqual(run.call_args.kwargs["latest_link"], latest.absolute())
            self.assertNotEqual(run.call_args.kwargs["latest_link"], previous.resolve())

    def test_uniform_sampler_excludes_terminal_and_freezes_drop_last_window(self) -> None:
        self.assertEqual(
            uniform_frame_indices(20, frame_count=4, end_exclusion=1).tolist(),
            [0, 6, 12, 18],
        )
        self.assertEqual(
            uniform_frame_indices(
                20, frame_count=4, end_exclusion=1, end_fraction=0.8
            ).tolist(),
            [0, 5, 9, 14],
        )
        with self.assertRaisesRegex(VideoInformationProbeError, "enough frames"):
            uniform_frame_indices(4, frame_count=4, end_exclusion=1)

    def test_all_controls_reuse_one_clip_without_shape_or_dtype_leakage(self) -> None:
        clip = np.arange(16 * 2 * 2 * 3, dtype=np.uint8).reshape(16, 2, 2, 3)
        reversed_clip = derive_clip_condition(clip, "reversed", shuffle_seed=7)
        shuffled_a = derive_clip_condition(clip, "shuffled", shuffle_seed=7)
        shuffled_b = derive_clip_condition(clip, "shuffled", shuffle_seed=7)
        first = derive_clip_condition(clip, "first_frame", shuffle_seed=7)
        last = derive_clip_condition(clip, "last_frame", shuffle_seed=7)
        median = derive_clip_condition(
            clip, "static_temporal_median", shuffle_seed=7
        )

        np.testing.assert_array_equal(reversed_clip, clip[::-1])
        np.testing.assert_array_equal(shuffled_a, shuffled_b)
        self.assertFalse(np.array_equal(shuffled_a, clip))
        self.assertFalse(np.array_equal(shuffled_a, clip[::-1]))
        np.testing.assert_array_equal(first, np.repeat(clip[:1], 16, axis=0))
        np.testing.assert_array_equal(last, np.repeat(clip[-1:], 16, axis=0))
        np.testing.assert_array_equal(median[0], median[-1])
        for value in (reversed_clip, shuffled_a, first, last, median):
            self.assertEqual(value.shape, clip.shape)
            self.assertEqual(value.dtype, np.uint8)

    def test_fixed_dual_ridge_readout_generalizes_on_separable_features(self) -> None:
        support = np.array(
            [[-2.0, 0.0], [-1.5, 0.1], [2.0, 0.0], [1.5, -0.1]],
            dtype=np.float64,
        )
        support_labels = np.array([3, 3, 4, 4])
        model = fit_frozen_linear_probe(
            support, support_labels, task_ids=[3, 4], ridge_lambda=1.0
        )
        result = score_linear_probe(
            model,
            np.array([[-1.8, 0.2], [1.8, -0.2]], dtype=np.float64),
            np.array([3, 4]),
        )

        self.assertEqual(result["predictions"].tolist(), [3, 4])
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertTrue(np.all(result["signed_margins"] > 0))

    def test_temporal_moments_have_predeclared_reversal_and_static_behavior(self) -> None:
        frames = np.array(
            [[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]]],
            dtype=np.float64,
        )
        ordered = temporal_moment_descriptor(frames).reshape(1, 5, 2)
        reversed_value = temporal_moment_descriptor(frames[:, ::-1]).reshape(1, 5, 2)
        static = temporal_moment_descriptor(np.repeat(frames[:, :1], 4, axis=1)).reshape(1, 5, 2)

        np.testing.assert_allclose(reversed_value[:, 0], ordered[:, 0], atol=1e-7)
        np.testing.assert_allclose(reversed_value[:, 1], ordered[:, 2], atol=1e-7)
        np.testing.assert_allclose(reversed_value[:, 2], ordered[:, 1], atol=1e-7)
        np.testing.assert_allclose(reversed_value[:, 3], -ordered[:, 3], atol=1e-7)
        np.testing.assert_allclose(reversed_value[:, 4], -ordered[:, 4], atol=1e-7)
        np.testing.assert_allclose(static[:, 3:], 0.0, atol=1e-7)

    def test_feature_schedule_reuses_ordered_support_and_all_query_controls(self) -> None:
        spec = load_video_spec(self.config_path)
        records = [
            {
                "task_id": task,
                "demo_index": demo,
                "partition": "support" if demo < 24 else "query",
            }
            for task in (3, 4)
            for demo in range(48)
        ]
        descriptors = feature_descriptors(spec, {"records": records})

        self.assertEqual(len(descriptors), 384)
        self.assertEqual(sum(row["partition"] == "support" for row in descriptors), 48)
        self.assertEqual(sum(row["partition"] == "query" for row in descriptors), 336)
        self.assertEqual(
            {row["condition"] for row in descriptors if row["partition"] == "query"},
            set(spec["conditions"]),
        )

    def test_stratified_interval_is_deterministic_and_balanced(self) -> None:
        labels = np.array([3, 3, 3, 4, 4, 4])
        correct = np.array([True, True, False, True, False, False])
        first = stratified_accuracy_interval(
            correct, labels, task_ids=[3, 4], samples=500, seed=11
        )
        second = stratified_accuracy_interval(
            correct, labels, task_ids=[3, 4], samples=500, seed=11
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["point_estimate"], 0.5)
        self.assertLessEqual(first["lower"], first["point_estimate"])
        self.assertGreaterEqual(first["upper"], first["point_estimate"])

    def test_decision_separates_content_from_temporal_claims(self) -> None:
        spec = load_video_spec(self.config_path)
        metrics = {
            "ordered_full": {"balanced_accuracy": 0.90},
            "reversed": {"balanced_accuracy": 0.70},
            "shuffled": {"balanced_accuracy": 0.75},
            "first_frame": {"balanced_accuracy": 0.60},
            "last_frame": {"balanced_accuracy": 0.80},
            "static_temporal_median": {"balanced_accuracy": 0.65},
            "drop_last_20_percent": {"balanced_accuracy": 0.85},
            "bidirectional_query_pair_fraction": 0.85,
            "wrong_video_specificity": 0.90,
        }
        positive = decide_video_probe(spec, metrics)
        self.assertEqual(
            positive["status"], "source_video_content_and_temporal_signal_present"
        )
        self.assertFalse(positive["gate_decision_authorized"])

        metrics["shuffled"]["balanced_accuracy"] = 0.88
        temporal_negative = decide_video_probe(spec, metrics)
        self.assertEqual(
            temporal_negative["status"],
            "source_video_content_present_temporal_order_not_established",
        )


if __name__ == "__main__":
    unittest.main()
