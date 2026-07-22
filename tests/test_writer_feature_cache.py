from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.writer.data import WriterTaskAuthority
from ember.writer.feature_cache import (
    PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
    PI05_TASK_FEATURE_CACHE_SCHEMA,
    FeatureCacheError,
    FeatureCacheTask,
    WriterFeatureStore,
    balanced_task_assignments,
    extraction_contract_sha256,
    load_pi05_feature_cache_config,
    load_pi05_feature_tasks,
    load_task_cache,
    pool_pi05_visual_tokens,
    save_task_cache,
    select_pi05_language_tokens,
    write_task_record,
)
from ember.pi05_source_checkpoint import canonical_hash, sha256_file, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]


def _task(task_id: int, frames: int) -> FeatureCacheTask:
    authority = WriterTaskAuthority(task_id, f"task {task_id}", Path("unused"), 1)
    return FeatureCacheTask(task_id, authority.language, authority, "0" * 64, (frames,))


def test_lpt_schedule_cover_every_task_once() -> None:
    tasks = tuple(_task(index, frames) for index, frames in enumerate((9, 8, 7, 6, 5, 4, 3, 2)))
    assignments = balanced_task_assignments(tasks, 3)
    assigned = [task.task_id for rank_tasks in assignments for task in rank_tasks]
    assert sorted(assigned) == list(range(8))
    loads = [sum(task.frame_count for task in rank_tasks) for rank_tasks in assignments]
    assert max(loads) - min(loads) <= max(task.frame_count for task in tasks)



def _seal_store(
    root: Path, extraction: str, records: list[dict[str, object]]
) -> tuple[str, str]:
    contract = {"schema_version": "test", "extraction_sha256": extraction}
    contract["contract_sha256"] = canonical_hash(contract)
    write_json_atomic(root / "run_contract.json", contract)
    manifest = {
        "schema_version": PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "extraction_sha256": extraction,
        "task_count": len(records),
        "task_records": records,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(root / "cache_manifest.json", manifest)
    return (
        sha256_file(root / "run_contract.json"),
        sha256_file(root / "cache_manifest.json"),
    )


def test_task_cache_roundtrip_preserves_episode_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "task.safetensors"
    record = save_task_cache(
        path,
        language_features=torch.randn(3, 5),
        generic_language_features=torch.randn(4, 5),
        video_features=torch.randn(7, 2, 5),
        episode_offsets=torch.tensor([0, 2, 7]),
        demo_indices=torch.tensor([4, 9]),
        metadata={"schema_version": "test"},
    )
    cached = load_task_cache(path, expected_dim=5, expected_spatial_tokens=2)
    assert record["frames"] == 7 and record["episodes"] == 2
    assert cached.video_features.dtype == torch.bfloat16
    assert cached.generic_language_features.shape == (4, 5)
    assert cached.episode_offsets.tolist() == [0, 2, 7]
    assert cached.demo_indices.tolist() == [4, 9]


def test_feature_store_validates_manifest_and_evicts_lru(tmp_path: Path) -> None:
    extraction = "a" * 64
    records = []
    for task_id in (2, 5):
        tensor_path = tmp_path / "tasks" / f"task_{task_id:03d}.safetensors"
        file_record = save_task_cache(
            tensor_path,
            language_features=torch.randn(2, 5),
            generic_language_features=torch.randn(3, 5),
            video_features=torch.randn(4, 2, 5),
            episode_offsets=torch.tensor([0, 4]),
            demo_indices=torch.tensor([0]),
            metadata={"extraction_sha256": extraction},
        )
        record = {
            "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
            "task_id": task_id,
            "extraction_sha256": extraction,
            "file": file_record,
        }
        write_task_record(tmp_path, task_id, record)
        records.append(record)
    run_sha, manifest_sha = _seal_store(tmp_path, extraction, records)
    store = WriterFeatureStore(
        tmp_path,
        task_ids=(2, 5),
        expected_extraction_sha256=extraction,
        max_cached_tasks=1,
        expected_dim=5,
        expected_spatial_tokens=2,
        expected_run_contract_file_sha256=run_sha,
        expected_manifest_file_sha256=manifest_sha,
    )
    assert store.load(2).episode_offsets.tolist() == [0, 4]
    assert store.cached_task_ids == (2,)
    store.load(5)
    assert store.cached_task_ids == (5,)


def test_feature_store_keeps_language_and_one_video_authorities_separate(
    tmp_path: Path,
) -> None:
    extraction = "b" * 64
    records = []
    for task_id, language_value, video_value in ((2, 2.0, 20.0), (5, 5.0, 50.0)):
        tensor_path = tmp_path / "tasks" / f"task_{task_id:03d}.safetensors"
        file_record = save_task_cache(
            tensor_path,
            language_features=torch.full((2, 3), language_value),
            generic_language_features=torch.full((3, 3), 7.0),
            video_features=torch.tensor(
                [[video_value], [video_value + 1], [video_value + 2], [video_value + 3]]
            ).repeat(1, 3)[:, None],
            episode_offsets=torch.tensor([0, 1, 4]),
            demo_indices=torch.tensor([7, 9]),
            metadata={"extraction_sha256": extraction},
        )
        record = {
            "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
            "task_id": task_id,
            "extraction_sha256": extraction,
            "file": file_record,
        }
        write_task_record(tmp_path, task_id, record)
        records.append(record)
    run_sha, manifest_sha = _seal_store(tmp_path, extraction, records)
    store = WriterFeatureStore(
        tmp_path,
        task_ids=(2, 5),
        expected_extraction_sha256=extraction,
        max_cached_tasks=2,
        expected_dim=3,
        expected_spatial_tokens=1,
        expected_run_contract_file_sha256=run_sha,
        expected_manifest_file_sha256=manifest_sha,
    )
    wrong_video = store.load_one_video(
        language_task_id=2, video_task_id=5, demo_index=9
    )
    assert torch.all(wrong_video.language_features == 2)
    assert torch.all(wrong_video.generic_language_features == 7)
    assert wrong_video.video_features[:, 0, 0].tolist() == [51, 52, 53]
    assert wrong_video.episode_offsets.tolist() == [0, 3]


def _write_pi05_feature_authorities(tmp_path: Path) -> Path:
    authority_dir = tmp_path / "authority"
    authority_dir.mkdir()
    split_roles = ["train"] * 24 + ["validation"] * 8 + ["test"] * 8
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    tasks = [
        {
            "global_task_id": task_id,
            "suite": suites[task_id // 10],
            "task_id": task_id % 10,
            "split_role": split_role,
            "language": f"task {task_id}",
            "hdf5": {
                "relative_path": f"suite/task_{task_id}.hdf5",
                "bytes": 100 + task_id,
                "sha256": f"{task_id:064x}",
            },
            "demonstrations": {
                "count": 50,
                "steps": 100,
                "episode_lengths": [2] * 50,
            },
        }
        for task_id, split_role in enumerate(split_roles)
    ]
    files = {
        "target.json": {
            "schema_version": "ember_pi05_target_data_manifest_v1",
            "dataset": {
                "repository": "yifengzhu-hf/LIBERO-datasets",
                "revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
            },
            "summary": {"tasks": 40, "episodes": 2000},
            "tasks": tasks,
        },
        "evaluation.json": {},
        "tokenizer.json": {},
        "lora.json": {},
    }
    for name, value in files.items():
        write_json_atomic(authority_dir / name, value)
    config = {
        "schema_version": "ember_pi05_writer_feature_cache_v2",
        "authorities": {
            name: {
                "path": f"authority/{filename}",
                "sha256": sha256_file(authority_dir / filename),
            }
            for name, filename in {
                "target_data_manifest": "target.json",
                "evaluation_config": "evaluation.json",
                "tokenizer_manifest": "tokenizer.json",
                "lora_contract": "lora.json",
            }.items()
        },
        "protocol": {
            "dataset_repository": "yifengzhu-hf/LIBERO-datasets",
            "dataset_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
            "role_split_roles": {
                "development": ["train", "validation"],
            },
            "role_task_counts": {
                "development": 32,
            },
            "demo_count_per_task": 50,
        },
        "model": {
            "policy_type": "pi05",
            "source_checkpoint": "final_formal_raw_policy_only",
            "vision_owner": "policy.model.paligemma_with_expert.embed_image",
            "language_owner": "policy.model.paligemma_with_expert.embed_language_tokens",
            "forbidden_checkpoint": "pi05_libero",
        },
        "information_wall": {
            "authorized_video_split_roles": ["train", "validation"],
            "test_video_values_read": 0,
            "trajectory_action_state_reward_terminal_reads": 0,
            "writer_input": "pure task language plus exactly one action-hidden agentview video",
        },
        "features": {
            "camera_dataset": "obs/agentview_rgb",
            "camera_transform": "libero_opengl_rotate_180_chw_unit_float_v1",
            "model_preprocessing": "PI05Policy._preprocess_images_resize_with_pad_224_neg_one_to_one",
            "frame_batch_size_per_rank": 32,
            "vision_token_count": 256,
            "vision_feature_dim": 2048,
            "vision_spatial_grid_size": 4,
            "vision_spatial_tokens": 16,
            "vision_pooling": "fixed_4x4_grid_mean_over_projected_spatial_tokens_per_frame",
            "vision_normalization": "none_after_pi05_projection",
            "language_feature_dim": 2048,
            "language_max_tokens": 64,
            "observed_target40_max_language_tokens": 23,
            "language_prompt": "Task: {cleaned_task}\n",
            "generic_writer_language": "perform the demonstrated task",
            "language_normalization": "none_after_pi05_embedding",
            "stored_dtype": "bfloat16",
            "preserve_episode_order_and_boundaries": True,
            "writer_invocation_video_count": 1,
        },
        "parallel": {
            "world_size": 8,
            "policy_processes_per_gpu": 1,
            "gpu0_extra_cuda_roles": 0,
            "assignment": "deterministic_lpt_by_manifest_frame_count",
            "task_level_atomic_resume": True,
        },
        "profile": {
            "status": "pending_source_base",
            "candidate_frame_batch_size_per_rank": 32,
            "selection_metric": "valid cached frames per second with zero OOM and exact tensor contract",
        },
    }
    path = tmp_path / "pi05_feature_cache.json"
    write_json_atomic(path, config)
    return path


def test_pi05_feature_config_and_development_role_are_sealed(tmp_path: Path) -> None:
    path = _write_pi05_feature_authorities(tmp_path)
    config = load_pi05_feature_cache_config(path, tmp_path)
    tasks = load_pi05_feature_tasks(
        config, tmp_path, tmp_path / "data", role="development"
    )
    assert len(tasks) == 32
    assert [task.task_id for task in tasks] == list(range(32))
    assert {task.split_role for task in tasks} == {"train", "validation"}
    assert tasks[0].authority.expected_sha256 == "0" * 64
    assert tasks[0].authority.path == tmp_path / "data/suite/task_0.hdf5"
    with pytest.raises(FeatureCacheError, match="unsupported PI05 feature-cache role"):
        load_pi05_feature_tasks(config, tmp_path, tmp_path / "data", role="test")


def test_pi05_feature_projection_adds_no_hidden_scaling() -> None:
    visual = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    pooled = pool_pi05_visual_tokens(
        visual, expected_tokens=4, expected_dim=3, spatial_grid_size=2
    )
    assert torch.equal(pooled, visual)

    language = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    mask = torch.tensor([[True, True, False, True, False, False]])
    selected = select_pi05_language_tokens(language, mask, expected_dim=3)
    assert torch.equal(selected, language[0, [0, 1, 3]])


def test_pi05_extraction_hash_covers_code_runtime_and_hdf5_authority() -> None:
    contract = {
        "schema_version": "pi05-test",
        "config_sha256": "a" * 64,
        "git": {"commit": "b" * 40},
        "authorities": {"target": {"sha256": "c" * 64}},
        "source": {
            "source_run_contract_sha256": "d" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "optimizer_step": 30_000,
            "source_run_summary_sha256": "f" * 64,
            "source_training_commit": "1" * 40,
            "source_base_config_sha256": "8" * 64,
            "source_authority_hashes": {"normalization": "9" * 64},
            "model_files": [{"path": "model.safetensors", "sha256": "2" * 64}],
        },
        "policy_files": {"model": {"sha256": "2" * 64, "bytes": 3}},
        "tokenizer": {"sha256": "3" * 64, "manifest_sha256": "4" * 64},
        "task_ids": [0],
        "tasks": [{"global_task_id": 0, "hdf5_sha256": "5" * 64}],
        "demo_indices": [0, 1],
        "features": {"vision_feature_dim": 2048},
        "runtime_versions": {"torch": "2.test"},
    }
    original = extraction_contract_sha256(contract)
    for key, value in (
        ("git", {"commit": "6" * 40}),
        ("runtime_versions", {"torch": "changed"}),
        ("tasks", [{"global_task_id": 0, "hdf5_sha256": "7" * 64}]),
    ):
        changed = dict(contract)
        changed[key] = value
        assert extraction_contract_sha256(changed) != original

    changed = dict(contract)
    changed["source"] = {
        **contract["source"],
        "source_authority_hashes": {"normalization": "0" * 64},
    }
    assert extraction_contract_sha256(changed) != original


def test_pi05_feature_store_may_open_a_sealed_manifest_subset(
    tmp_path: Path,
) -> None:
    extraction = "c" * 64
    records = []
    for task_id in (1, 2):
        tensor_path = tmp_path / "tasks" / f"task_{task_id:03d}.safetensors"
        file_record = save_task_cache(
            tensor_path,
            language_features=torch.randn(2, 4),
            generic_language_features=torch.randn(3, 4),
            video_features=torch.randn(3, 2, 4),
            episode_offsets=torch.tensor([0, 3]),
            demo_indices=torch.tensor([0]),
            metadata={"extraction_sha256": extraction},
        )
        record = {
            "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
            "task_id": task_id,
            "extraction_sha256": extraction,
            "file": file_record,
        }
        write_task_record(tmp_path, task_id, record)
        records.append(record)
    run_sha, manifest_sha = _seal_store(tmp_path, extraction, records)
    store = WriterFeatureStore(
        tmp_path,
        task_ids=(2,),
        expected_extraction_sha256=extraction,
        max_cached_tasks=1,
        expected_dim=4,
        expected_spatial_tokens=2,
        expected_run_contract_file_sha256=run_sha,
        expected_manifest_file_sha256=manifest_sha,
    )
    assert store.load(2).video_features.shape == (3, 2, 4)
