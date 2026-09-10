"""One-condition supervised FM chain-rule oracle through complete Writer and Meta."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ember.writer.supervised import SupervisedEngine


@pytest.mark.parametrize("conditions", [1, 2])
def test_fm_weights_once_and_replays_writer_and_meta_once(monkeypatch, conditions):
    meta = nn.Parameter(torch.tensor(2.0))

    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(3.0))
            self.calls = 0

        def forward(self, responses):
            self.calls += 1
            return {"w": self.weight * responses[0]}

    class Observer:
        device = torch.device("cpu")
        backward_calls = 0

        def responses(self, condition):
            return ((meta * 2).detach(),)

        def writer_arguments(self, condition):
            return ()

        def backward(self, condition, cotangents):
            self.backward_calls += 1
            (meta * 2).backward(cotangents[0])

    writer, observer = Writer(), Observer()
    engine = object.__new__(SupervisedEngine)
    engine.device = torch.device("cpu")
    engine.cache = SimpleNamespace(condition=lambda *args: None, hits=0, misses=0, bytes=0)
    engine.config = {"runtime": {"policy_microbatch": 8}, "data": {"conditions_per_task": conditions}}
    engine.runtime = SimpleNamespace(observer=observer, state=SimpleNamespace(writer=writer), policy=None, lora=None,
                                     processor=SimpleNamespace(training_batch=lambda batch: batch))
    engine.data = SimpleNamespace(action_batch=lambda *args, **kwargs: ({}, {
        "policy_rng_seed": 5, "policy_random_batch_size": 64,
        "action_demos": [16] * (64 // conditions), "action_frames": list(range(64 // conditions)),
    }))

    def gradient(policy, state, *args, **kwargs):
        assert kwargs["policy_random_batch_size"] == 64
        assert kwargs["policy_batch_offset"] == 64 - 64 // conditions
        return state["w"].square() / 2, {}, {"w": state["w"].detach().clone()}
    monkeypatch.setattr("ember.writer.supervised.functional_lora_loss_gradient", gradient)
    metric = engine.backward({"task": 0, "video_demos": (0,), "occurrence": 0, "query_seed": 7,
                              "query_offset": 64 - 64 // conditions, "query_count": 64 // conditions})
    # Generated w = writer_weight * (2 * meta) = 12; loss=w²/2 and task weight=1/4.
    torch.testing.assert_close(writer.weight.grad, torch.tensor(12.0 / conditions))
    torch.testing.assert_close(meta.grad, torch.tensor(18.0 / conditions))
    assert writer.calls == 2 and observer.backward_calls == 1
    assert metric["queries"] == 64 // conditions
    assert metric["condition_weight"] == 0.25 / conditions
    assert not any(key.startswith("rl_") for key in metric)
