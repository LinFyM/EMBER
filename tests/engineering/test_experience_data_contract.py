"""Real metadata and CPU queries for fixed36/held-label separation."""
from collections import Counter
from dataclasses import replace
from itertools import islice

import pytest
import torch

from ember.experience_compiler import contract
from ember.experience_compiler.data import QueryData, event_for_update
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json


@pytest.fixture(scope="module")
def tasks():
    return contract.training_tasks()


def test_continuous_equal_weight_events_and_cross_episode_time_intervals(tasks):
    events = [event for update in range(128, 182) for event in event_for_update(tasks, update)]
    assert Counter(event.task_id for event in events) == {task: 6 for task in contract.TASKS36}
    assert sum(event.masked_experience for event in events) == 27
    assert events[:4] == event_for_update(tasks, 128)
    assert len({event.condition_id for event in events}) == 216
    for event in events:
        counts = Counter(demo for demo, _ in event.queries28)
        assert len(counts) == 7 and set(counts.values()) == {4}
        assert event.teacher_demo not in counts and len(set(event.query_states2)) == 2
        for index, (demo, frame) in enumerate(event.queries28):
            available = tasks[event.task_id].episode_lengths[demo] - 1
            interval = index % 4
            assert available * interval // 4 <= frame < available * (interval + 1) // 4
        assert event.as_dict()["task_id"] == event.task_id
    first_task_teachers = [event_for_update(tasks, 9 * visit)[0].teacher_demo for visit in range(50)]
    assert set(first_task_teachers) == set(range(50))


def test_state_stream_is_disjoint_deterministic_and_not_capped_by_pool(tasks):
    event = event_for_update(tasks, 128)[0]
    excluded = (*event.query_states2, 32, 33, 34)
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


def test_real_train_query_and_RGB_reader_with_source_only_processor(tasks):
    data = QueryData(contract.ASSET_ROOT)
    event = data.events(0)[0]
    normalization = read_json(contract.ASSET_ROOT / contract.SOURCE["normalization"])["stats"]
    processor = Pi05LiberoProcessor(normalization, contract.ASSET_ROOT / contract.SOURCE["tokenizer"],
                                   max_length=200, device="cpu")
    try:
        assert set(data.queries.authorities) == set(contract.TASKS36)
        video = data.videos.load(event.task_id, event.teacher_demo)
        assert video.frames.ndim == 5 and video.frames.shape[1:3] == (2, 3)
        assert video.frame_indices[0] == 0 and video.frame_indices[-1] == video.raw_frame_count - 1
        batch = data.query_batch(event, processor)
        assert batch["action"].shape == (28, 50, 7) and torch.isfinite(batch["action"]).all()
        assert not {"task_id", "demo_index", "frame_index", "filename"} & set(batch)
        with pytest.raises(ValueError, match="fixed36"):
            data.query_batch(replace(event, task_id=3), processor)
        data.next_update = 129
        state = data.state_dict()
        data.next_update = 0
        data.load_state_dict(state)
        assert data.events() == event_for_update(tasks, 129)
        with pytest.raises(ValueError, match="identity"):
            data.load_state_dict({**state, "seed": 1})
    finally:
        data.close()
