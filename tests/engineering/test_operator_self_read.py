"""Small composite-credit oracle and the sealed shared self-read identity."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state
from ember.operator_writer import joint_training, run
from ember.operator_writer.model import OperatorReadWrite
from ember.operator_writer.data import FormalData
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract


def small_writer(mode):
    root = Path(__file__).resolve().parents[2]
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / "configs/pi05_lora_v1.json"), rank=128)
    contract = replace(contract, targets=tuple(LoRATarget(target.name, 4, 3) for target in contract.targets))
    return OperatorReadWrite(contract, identity_lora_state(contract), mode), contract


def test_same_Context_initialization_and_sealed_fresh_event_identity():
    original, _ = small_writer("context")
    before = torch.random.get_rng_state()
    candidate, _ = small_writer("self_read")
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in candidate.parameters())
    for name in ("common.values.0", "writes.0.p.weight", "writes.37.o.weight", "value_context.wh.weight"):
        torch.testing.assert_close(dict(original.named_parameters())[name], dict(candidate.named_parameters())[name])
    spec = run.specification(run.SELF_READ_SPEC_PATH)
    parent = run.specification(run.CONTEXT_SPEC_PATH)
    assert spec["operator"]["self_conditioned_native"] == joint_training.SELF_READ
    for key in ("source", "events", "optimization"):
        assert spec[key] == parent[key]
    assert spec["execution"]["world_sizes"] == list(range(1, 7))
    assert joint_training.settings(spec)[1] == "self_read"
    args = SimpleNamespace(mode="self_read", attempt="fresh", resume=None, pilot_arm=None,
                           microbatch=28, frame_chunk=8, stop_after_macro=None)
    run.validate_train_request(spec, args)
    args.resume = Path("/not/an/owned/self_read/ECP")
    with pytest.raises(ValueError, match="fresh identity"):
        run.validate_train_request(spec, args)
    asset = Path("/data1/user/ymdai/projects/EMBER")
    old, new = FormalData(asset, parent, query_labels=False), FormalData(asset, spec, query_labels=False)
    try:
        for step in range(450):
            assert old.tasks_for_step(step) == new.tasks_for_step(step)
            for task in new.tasks_for_step(step):
                assert old.event(step, task) == new.event(step, task)
    finally:
        old.close(); new.close()


def test_actual_compile_composite_credit_unique_output_and_unchanged_public(monkeypatch):
    torch.manual_seed(23)
    writer, contract = small_writer("self_read")
    first = writer.names[0]
    with torch.no_grad():
        writer.writes[0].o.weight.normal_(std=.04)
        writer.writes[0].u.weight.normal_(std=.04)
        writer.public_state()[first + LORA_B_SUFFIX].normal_(std=.02)
    inputs = {name: torch.randn(3, 50, 4) for name in writer.names}
    hidden, modulation = torch.randn(3, 50, 1024), torch.randn(3, 50, 1024)
    reads = []
    def native(policy, state, probe, condition, names, **kwargs):
        reads.append(state)
        effective = state[first + LORA_B_SUFFIX] @ state[first + LORA_A_SUFFIX]
        signal = effective.square().sum().tanh()
        return {name: value + signal * .1 for name, value in inputs.items()}, hidden + signal * modulation
    monkeypatch.setattr(run, "read_native_video", native)
    monkeypatch.setattr(run.Runtime, "restore_identity", lambda self: None)
    runtime = run.Runtime(None, writer, None, None, contract, {}, torch.device("cpu"), {})
    base = writer.public_state()
    original_B = base[first + LORA_B_SUFFIX].detach().clone()
    condition = (None, [0, 5, 11], None, None)
    state, recorded = runtime.compile(condition, retain_native=True)
    assert len(reads) == len(recorded["passes"]) == 2
    assert reads[0][first + LORA_B_SUFFIX] is base[first + LORA_B_SUFFIX]
    assert reads[1][first + LORA_B_SUFFIX] is recorded["passes"][0]["state"][first + LORA_B_SUFFIX]
    torch.testing.assert_close(base[first + LORA_B_SUFFIX], original_B)
    assert state is recorded["passes"][1]["state"]
    assert (recorded["passes"][0]["h"] - recorded["passes"][1]["h"]).norm() > 0
    loss = state[first + LORA_B_SUFFIX].square().sum() + state[first + LORA_A_SUFFIX].square().mean()
    parameter = writer.writes[0].o.weight
    actual = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    # Independent direct-autograd composition and chain-rule decomposition.
    x0, h0 = native(None, base, writer.probe, condition, writer.names)
    first_state = writer(x0, h0, frame_indices=condition[1])
    detached = {name: value.detach().requires_grad_() for name, value in first_state.items()}
    x1, h1 = native(None, detached, writer.probe, condition, writer.names)
    final = writer(x1, h1, frame_indices=condition[1])
    oracle_loss = final[first + LORA_B_SUFFIX].square().sum() + final[first + LORA_A_SUFFIX].square().mean()
    direct, *cotangents = torch.autograd.grad(oracle_loss, (parameter, *detached.values()), allow_unused=True, retain_graph=True)
    used = [(first_state[name], cotangent) for name, cotangent in zip(detached, cotangents) if cotangent is not None]
    indirect = torch.autograd.grad(tuple(x for x, _ in used), parameter,
                                  grad_outputs=tuple(g for _, g in used), retain_graph=True)[0]
    torch.testing.assert_close(actual, direct + indirect, rtol=3e-4, atol=3e-6)
    assert indirect.norm() > 0
    loss.backward()
    assert recorded["passes"][0]["state"][first + LORA_B_SUFFIX].grad.norm() > 0
    assert recorded["passes"][0]["h"].grad.norm() > 0
    assert writer.public_state()[first + LORA_B_SUFFIX].grad.norm() > 0
    torch.testing.assert_close(base[first + LORA_B_SUFFIX], original_B)
