"""One-condition chain-rule oracle across FM, episode credit and native Meta."""

import random
from types import SimpleNamespace

import torch
from torch import nn

from ember.writer.joint import JointUpdateEngine, TrustEvidence
from ember.writer.rl_math import DecisionReservoir
from ember.writer.rollout import EpisodeTrace, _EnvironmentRNG, recorded_flow_batches


def test_joint_credit_weights_once_and_replays_writer_and_meta_once(monkeypatch):
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
    episodes = []
    for index, count in enumerate((20, 10, 5, 3)):
        reservoir = DecisionReservoir(random.Random(index))
        for _ in range(count):
            reservoir.add({"old_mean": torch.full((35,), 12.0), "z": torch.full((35,), 12.2),
                           "flow_batch_size": index + 1})
        episodes.append(EpisodeTrace({"trust": index}, reservoir, index == 0, count * 5))
    engine = object.__new__(JointUpdateEngine)
    engine.device, engine.precision = torch.device("cpu"), torch.eye(35)
    engine.cache = SimpleNamespace(condition=lambda *args: None)
    engine.config = {"runtime": {"policy_microbatch": 8, "rl_microbatch": 4}}
    engine.runtime = SimpleNamespace(observer=observer, state=SimpleNamespace(writer=writer), policy=None, lora=None,
                                     processor=SimpleNamespace(training_batch=lambda batch: batch))
    engine.data = SimpleNamespace(action_batch=lambda *args, **kwargs: ({}, {
        "policy_rng_seed": 5, "action_demos": [16] * 64, "action_frames": list(range(64)),
    }))

    def collect(task, state, seeds):
        assert not state["w"].requires_grad
        assert float(state["w"]) == 12
        assert writer.weight.grad is None and meta.grad is None
        return episodes

    engine.rollouts = SimpleNamespace(collect=collect)
    monkeypatch.setattr("ember.writer.joint.functional_lora_loss_gradient", lambda policy, state, *args, **kwargs:
                        (state["w"].square() / 2, {}, {"w": state["w"].detach().clone()}))
    monkeypatch.setattr("ember.writer.joint.decision_batch", lambda records, device:
                        ({}, torch.zeros(len(records), 50, 32)))
    monkeypatch.setattr("ember.writer.joint.flow_mean_lora_gradient", lambda policy, state, contract, batch, noise, cotangent:
                        {"w": cotangent.sum()})
    metric, evidence = engine.backward({"task": 0, "video_demos": (0,), "occurrence": 0,
                                       "query_seed": 7, "episodes": [{}] * 4})
    # m=w in this oracle, L_FM=w²/2. Full decision sum, task1/4, episode1/4.
    advantage_weighted_decisions = 20 - (10 + 5 + 3) / 3
    expected = 12 / 4 - 0.1 / 16 * advantage_weighted_decisions * 35 * (float(torch.tensor(12.2)) - 12)
    torch.testing.assert_close(writer.weight.grad, torch.tensor(expected * 4))
    torch.testing.assert_close(meta.grad, torch.tensor(expected * 6))
    assert writer.calls == 2 and observer.backward_calls == 1
    assert metric["rl_mixed_group"] and metric["rl_successes"] == 1
    assert len(evidence.records) == 15  # Three episodes >=4 decisions; one has only three.


def test_trust_replays_collected_batch_shape_without_weighting_padding(monkeypatch):
    sizes = [4, 1, 3, 2, 4, 3, 2]
    records = [{"flow_batch_size": size, "observation": {"x": torch.zeros(1, 1)},
                "noise": torch.full((1, 50, 32), float(index)),
                "old_mean": torch.full((35,), float(index + size))}
               for index, size in enumerate(sizes)]
    batches = list(recorded_flow_batches(records))
    assert sorted(position for _, positions in batches for position in positions) == list(range(7))
    assert all(len(chunk) == record["flow_batch_size"] for chunk, _ in batches for record in chunk)
    engine = object.__new__(JointUpdateEngine)
    engine.device, engine.precision = torch.device("cpu"), torch.eye(35)
    engine.config = {"runtime": {"trust_microbatch": 4}}
    engine.runtime = SimpleNamespace(
        observer=SimpleNamespace(responses=lambda condition: (), writer_arguments=lambda condition: ()),
        state=SimpleNamespace(writer=lambda *args: {}), policy=None, lora=None,
    )
    # Numerical shape is deliberately observable, as in the real BF16 failure.
    monkeypatch.setattr("ember.writer.joint.flow_actions", lambda policy, state, contract, batch, noise:
                        noise + len(noise))
    evidence = TrustEvidence(21, None, records, [0, 1, 2, 3, 0, 1, 2])
    assert engine.trust_score(evidence) == 0.0


def test_environment_rng_is_independent_of_lane_interleaving():
    import numpy as np

    def draw():
        return random.random(), float(np.random.random())

    np.random.seed(19)
    random.seed(19)
    expected_global = draw()
    np.random.seed(19)
    random.seed(19)
    first, second = _EnvironmentRNG(4), _EnvironmentRNG(9)
    observed = []
    for _ in range(5):
        observed.append(first.call(draw))
        second.call(draw)
    isolated = _EnvironmentRNG(4)
    assert observed == [isolated.call(draw) for _ in range(5)]
    assert draw() == expected_global
