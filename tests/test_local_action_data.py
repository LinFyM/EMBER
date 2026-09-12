"""Local inverse-action pairs retain raw timing and independent sampling roles."""

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ember.writer import learning_data
from ember.writer.data import WriterTaskAuthority
from ember.writer.errors import WriterModelError
from ember.writer.learning_data import LearningTask, WriterTrainingData


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    path = tmp_path / "actions.hdf5"
    lengths = tuple(31 + demo % 5 for demo in range(50))
    raw = {}
    with h5py.File(path, "w") as handle:
        for demo, length in enumerate(lengths):
            group = handle.create_group(f"data/demo_{demo}")
            frames = (np.arange(length * 2 * 3 * 3).reshape(length, 2, 3, 3) + demo).astype(np.uint8)
            actions = demo * 1000 + np.arange(length)[:, None] * 10 + np.arange(7)[None]
            group.create_dataset("actions", data=actions.astype(np.float64))
            obs = group.create_group("obs")
            obs.create_dataset("agentview_rgb", data=frames)
            obs.create_dataset("eye_in_hand_rgb", data=255 - frames)
            obs.create_dataset("ee_states", data=np.zeros((length, 6)))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2)))
            raw[demo] = (frames, actions)
    tasks = {
        task: LearningTask(WriterTaskAuthority(task, f"task {task}", path, path.stat().st_size),
                           f"suite{index}", 0, lengths)
        for index, task in enumerate((0, 12, 20, 34))
    }
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_args, **_kwargs: tasks)
    config = json.loads((ROOT / "configs/pi05_video_functional.json").read_text())["data"]
    instances = []

    def create(*, seed=None, camera_view="agentview"):
        selected = dict(config)
        if seed is not None:
            selected["seed"] = seed
        data = WriterTrainingData(ROOT, selected, camera_view=camera_view)
        instances.append(data)
        return data

    yield create, raw
    for data in instances:
        data.close()


@pytest.mark.parametrize("camera", ["agentview", "eye_in_hand", "dual"])
@pytest.mark.parametrize("at_end", [False, True])
def test_local_clip_uses_post_action_slice_and_real_endpoint(local_data, camera, at_end):
    create, raw = local_data
    data = create(camera_view=camera)
    pixels, actions = raw[16]
    start = len(actions) - 16 if at_end else 0
    frames, indices, labels = data.queries.local_action_clip(0, 16, start, camera_view=camera)
    selected = [start, start + 5, start + 10, start + 15]
    expected = pixels[selected, ::-1, ::-1].transpose(0, 3, 1, 2)
    if camera == "eye_in_hand":
        expected = 255 - expected
    elif camera == "dual":
        expected = np.stack((expected, 255 - expected), axis=1)
    np.testing.assert_array_equal(frames, expected)
    assert indices.tolist() == selected
    np.testing.assert_array_equal(labels, actions[start + 1:start + 16])
    assert labels.shape == (15, 7) and labels.dtype == np.float32


def test_local_clip_reads_only_rgb_and_action_values(local_data, monkeypatch):
    create, _ = local_data
    data = create(camera_view="dual")
    reads = []
    original = h5py.Dataset.__getitem__

    def audited(dataset, key):
        assert dataset.name.endswith(("/actions", "/agentview_rgb", "/eye_in_hand_rgb"))
        reads.append(dataset.name)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", audited)
    frames, indices, actions, trace = data.local_action_clip(0, 5)
    assert len(reads) == 3 and all(f"demo_{trace['local_action_demo']}/" in name for name in reads)
    assert len(frames) == len(indices) == 1 and frames[0].shape[:2] == (4, 2)
    assert all(value.device.type == "cpu" for value in (*frames, *indices, actions))
    assert frames[0].dtype == torch.uint8 and indices[0].dtype == torch.int64
    assert actions.dtype == torch.float32
    reads.clear()
    data.videos.load(0, 0)
    assert len(reads) == 2 and all("/obs/" in name for name in reads)


def test_local_clip_refuses_unowned_episodes_and_incomplete_intervals(local_data, monkeypatch):
    create, raw = local_data
    data = create()
    monkeypatch.setattr(data.queries, "_handle", lambda *_: pytest.fail("invalid clip opened HDF5"))
    for task, demo in ((999, 16), (0, 0), (0, 42), (0, 46)):
        with pytest.raises(WriterModelError, match="authority"):
            data.queries.local_action_clip(task, demo, 0)
    for start in (-1, len(raw[16][1]) - 15):
        with pytest.raises(WriterModelError, match="without padding"):
            data.queries.local_action_clip(0, 16, start)
    for occurrence in (-1, 16):
        with pytest.raises(ValueError, match="authority|occurrence"):
            data.local_action_clip(0, occurrence, diagnostic=True)
    data.action_pool = (0,)
    with pytest.raises(ValueError, match="roles"):
        data.local_action_clip(0, 0)


def test_local_sampling_preserves_main_draws_queries_and_resume(local_data):
    create, _ = local_data
    data, control = create(), create()
    assert data.next_iteration() == control.next_iteration()
    saved = deepcopy(data.sampler_state())
    for occurrence in range(16):
        _, _, _, trace = data.local_action_clip(0, occurrence)
        assert 16 <= trace["local_action_demo"] <= 41 and not trace["local_diagnostic"]
        data.local_action_clip(0, occurrence, diagnostic=True)
    assert data.sampler_state() == saved
    expected = [control.next_iteration() for _ in range(3)]
    assert [data.next_iteration() for _ in range(3)] == expected
    args = dict(task=0, occurrence=2, demos=(0,), query_seed=123)
    batch, trace = data.action_batch(**args)
    baseline_batch, baseline_trace = control.action_batch(**args)
    assert trace == baseline_trace
    for key in ("action", "frame_index", "demo_index", "observation.state"):
        torch.testing.assert_close(batch[key], baseline_batch[key])
    first = data.local_action_clip(0, 7)
    data.restore_sampler(saved)
    repeat = data.local_action_clip(0, 7)
    assert first[3] == repeat[3]
    torch.testing.assert_close(first[2], repeat[2])
    assert [data.next_iteration() for _ in range(3)] == expected


def test_local_diagnostic_is_fixed_across_training_seeds_and_four_per_episode(local_data):
    create, raw = local_data
    data, independent = create(seed=1), create(seed=987)
    traces = []
    for occurrence in range(16):
        frames, indices, actions, trace = data.local_action_clip(0, occurrence, diagnostic=True)
        other = independent.local_action_clip(0, occurrence, diagnostic=True)
        assert trace == other[3]
        assert trace["local_action_demo"] == 42 + occurrence // 4
        assert trace["local_diagnostic"] and trace["local_frame_indices"] == indices[0].tolist()
        start = trace["local_start_frame"]
        assert trace["local_action_start"] == start + 1 and trace["local_action_stop"] == start + 16
        expected = raw[trace["local_action_demo"]][1][start + 1:start + 16]
        np.testing.assert_array_equal(actions.numpy(), expected)
        torch.testing.assert_close(frames[0], other[0][0])
        traces.append(trace)
    assert len({trace["local_flow_seed"] for trace in traces}) == 16
