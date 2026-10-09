"""Real metadata and CPU queries for fixed36/held-label separation."""
from collections import Counter
from dataclasses import replace
from itertools import islice
import json

import pytest
import torch

from ember.experience_compiler import contract
from ember.experience_compiler.data import QueryData, collection_conditions, event_for_update
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideoStore


@pytest.fixture(scope="module")
def tasks():
    return contract.training_tasks()


@pytest.fixture(scope="module")
def recorded_pools(tmp_path_factory):
    """Sampler-facing metadata only; this fixture does not simulate practice."""
    root = tmp_path_factory.mktemp("registered-pools")
    pools, paths = {}, {}
    for pool in ("pool0", "refresh180"):
        conditions = collection_conditions(pool)
        for condition in conditions:
            condition["record_path"] = str(root / condition["condition_id"])
            condition["events"] = [
                {"endpoint": 2, "incoming": "MT", "behavior_version": pool},
                {"endpoint": 7, "incoming": "MT" if pool == "pool0" else "incoming_001.safetensors",
                 "behavior_version": pool},
            ]
        manifest = {"schema_version": contract.EVENT_SCHEMA, "pool": pool, "complete": True,
                    "version": f"cpu-metadata-{pool}", "conditions": conditions}
        path = root / f"{pool}.json"
        path.write_text(json.dumps(manifest))
        paths[pool] = path
        pools[pool] = {"version": manifest["version"], "by_task": {
            task: [c for c in conditions if c["task_id"] == task] for task in contract.TASKS36}}
    return pools, paths


def test_task_weights_pool_refresh_and_experience_mask_balance(tasks, recorded_pools):
    pools, _ = recorded_pools
    events = [event for update in range(360) for event in event_for_update(tasks, pools, update)]
    assert len(events) == 360 * 4
    assert events[:4] == event_for_update(tasks, pools, 0)
    assert Counter(event.task_id for event in events) == {task: 40 for task in contract.TASKS36}
    for start in range(0, len(events), 36):
        assert Counter(event.task_id for event in events[start:start + 36]) == {
            task: 1 for task in contract.TASKS36}
    for task in contract.TASKS36:
        first = [e for e in events[:180 * 4] if e.task_id == task]
        second = [e for e in events[180 * 4:] if e.task_id == task]
        assert Counter(e.pool for e in first) == {"pool0": 20}
        assert Counter(e.pool for e in second) == {"pool0": 10, "refresh180": 10}
        assert sum(e.masked_experience for e in first + second) == 5
        old, fresh = (pools[pool]["by_task"][task] for pool in ("pool0", "refresh180"))
        assert len(old) == 4 and len(fresh) == 2
        assert not {c["teacher_demo"] for c in old} & {c["teacher_demo"] for c in fresh}
    assert all(sum(e.masked_experience for e in events[start:start + 8]) == 1
               for start in range(0, len(events), 8))


def test_events_preserve_actual_incoming_and_cross_episode_query_authority(tasks, recorded_pools):
    pools, _ = recorded_pools
    events = [event for update in range(360) for event in event_for_update(tasks, pools, update)]
    for event in events:
        candidates = pools[event.pool]["by_task"][event.task_id]
        condition = next(c for c in candidates if c["condition_id"] == event.condition_id)
        assert event.teacher_demo == condition["teacher_demo"]
        assert event.record_path == condition["record_path"]
        assert any((event.endpoint, event.incoming, event.behavior_version) ==
                   (e["endpoint"], e["incoming"], e["behavior_version"]) for e in condition["events"])
        counts = Counter(demo for demo, _ in event.queries28)
        assert len(counts) == 7 and set(counts.values()) == {4}
        assert event.teacher_demo not in counts
        for index, (demo, frame) in enumerate(event.queries28):
            available = tasks[event.task_id].episode_lengths[demo] - 1
            interval = index % 4
            assert available * interval // 4 <= frame < available * (interval + 1) // 4
        assert event.as_dict()["task_id"] == event.task_id


def test_sampler_resume_attaches_refresh_only_at_registered180_boundary(tasks, recorded_pools):
    pools, paths = recorded_pools
    sampler = QueryData.__new__(QueryData)
    sampler.tasks, sampler.pools, sampler.next_update = tasks, {}, 180
    sampler.attach_pool(paths["pool0"])
    old = sampler.state_dict()
    sampler.attach_pool(paths["refresh180"])
    sampler.load_state_dict(old)
    assert sampler.events() == event_for_update(tasks, pools, 180)
    with pytest.raises(ValueError, match="registered180 boundary"):
        sampler.load_state_dict({**old, "next_update": 179})
    bad_version = {**old, "pool_versions": {"pool0": "different-behavior"}}
    with pytest.raises(ValueError, match="pool version"):
        sampler.load_state_dict(bad_version)
    state = sampler.state_dict()
    sampler.next_update = 0
    sampler.load_state_dict(state)
    assert sampler.events() == event_for_update(tasks, pools, 180)
    with pytest.raises(ValueError, match="authority"):
        sampler.load_state_dict({**state, "seed": 1})


def test_state_stream_is_disjoint_deterministic_and_not_capped_by_pool(tasks, recorded_pools):
    event = event_for_update(tasks, recorded_pools[0], 0)[0]
    excluded = (32, 33, 34)
    count = 50 - len(set(excluded))
    first = list(islice(contract.state_stream(event, excluded), count * 3 + 1))
    assert first == list(islice(contract.state_stream(event.as_dict(), excluded), len(first)))
    assert not set(first) & set(excluded)
    assert set(first[:count]) == set(range(50)) - set(excluded)
    assert set(first[count:2 * count]) == set(first[:count])


def test_panel_uses_original_manifest_order_and_formal400_matches_original(tasks):
    panel = contract.panel_contract(tasks)
    assert panel["task_ids"] == [0, 1, 12, 13, 20, 21, 32, 34]
    assert len(panel["conditions"]) == 16 and panel["final_state_ids"] == [32, 33, 34]
    reversed_order = contract.panel_contract(dict(reversed(tuple(tasks.items()))))
    assert reversed_order == panel
    rows = contract.formal400_mapping()
    assert len(rows) == 400 and len({row["condition_id"] for row in rows}) == 400
    assert {row["video_schedule_seed"] for row in rows} == {7}
    for task_id in (3, 6, 11, 16, 23, 26, 31, 39):
        subset = [row for row in rows if row["task_id"] == task_id]
        assert {row["teacher_demo"] for row in subset} == set(range(50))
        assert {row["init_state_id"] for row in subset} == set(range(50))


def test_environment_metadata_restores_full50_without_old_scene_or_launch_fields():
    learning, formal = contract.learning_environment(), contract.formal_environment()
    allowed = {"environment", "libero_paths", "policy", "rng", "tasks", "role"}
    assert set(learning) == allowed
    assert set(formal) == allowed | {"operator_read_write_scene"}
    original = read_json(contract.T_RESULTS.with_name("run_contract.json"))
    assert formal["operator_read_write_scene"] == original["operator_read_write_scene"]
    assert tuple(row["global_task_id"] for row in learning["tasks"]) == contract.TASKS36
    assert len(formal["tasks"]) == 8
    for environment in (learning, formal):
        for task in environment["tasks"]:
            assert task["init_state_ids"] == list(range(50))
            assert task["installed_init_state_count"] == 50
        assert environment["policy"]["replan_steps"] == 5
        assert environment["environment"]["dummy_settling_steps"] == 10
    assert contract.MT_PATH.is_file() and contract.MT_RESULTS.is_file() and contract.T_RESULTS.is_file()


def test_real_validation_teacher_store_uses_only_fixed_held_metadata_and_seals_Test():
    from ember.experience_compiler.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.asset_root, runtime.stores = contract.ASSET_ROOT, {}
    tasks, store = runtime._store('validation')
    try:
        assert set(tasks) == {3, 6, 11, 16, 23, 26, 31, 39}
        assert set(store.authorities) == set(tasks)
        assert store.frame_stride == 5 and store.camera_view == 'dual'
        assert not store._handles  # Metadata admission opens no held episode labels.
        assert runtime._store('validation') == (tasks, store)
        with pytest.raises(ValueError, match='Test is sealed'):
            runtime._store('test')
    finally:
        store.close()


def test_real_train_query_and_RGB_reader_with_source_only_processor(tasks, recorded_pools):
    data = QueryData(contract.ASSET_ROOT, pool_paths=tuple(recorded_pools[1].values()))
    videos = RawTeacherVideoStore(tuple(t.authority for t in tasks.values()), frame_stride=5, camera_view="dual")
    event = data.events(0)[0]
    normalization = read_json(contract.ASSET_ROOT / contract.SOURCE["normalization"])["stats"]
    processor = Pi05LiberoProcessor(normalization, contract.ASSET_ROOT / contract.SOURCE["tokenizer"],
                                   max_length=200, device="cpu")
    try:
        assert set(data.queries.authorities) == set(contract.TASKS36)
        video = videos.load(event.task_id, event.teacher_demo)
        assert video.frames.ndim == 5 and video.frames.shape[1:3] == (2, 3)
        assert video.frame_indices[0] == 0 and video.frame_indices[-1] == video.raw_frame_count - 1
        batch = data.query_batch(event, processor)
        assert batch["action"].shape == (28, 50, 7) and torch.isfinite(batch["action"]).all()
        assert not {"task_id", "demo_index", "frame_index", "filename"} & set(batch)
        with pytest.raises(ValueError, match="allowed task"):
            data.query_batch(replace(event, task_id=3), processor)
        with pytest.raises(ValueError, match="nonteacher"):
            data.raw_query_batch(replace(event, teacher_demo=event.queries28[0][0]))
        data.next_update = 129
        state = data.state_dict()
        data.next_update = 0
        data.load_state_dict(state)
        assert data.events() == event_for_update(tasks, data.pools, 129)
        with pytest.raises(ValueError, match="authority"):
            data.load_state_dict({**state, "seed": 1})
    finally:
        data.close()
        videos.close()
