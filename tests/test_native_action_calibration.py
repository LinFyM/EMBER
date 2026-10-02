"""Targeted scientific-consumer checks for the bounded task-only diagnostic."""
import copy

import numpy as np
import torch

from ember.operator_writer.native_action_calibration import Calibration, fresh_pair, paired_forward, sample_events, FIT


def test_zero_output_and_independent_initialization_preserves_rng():
    state=torch.get_rng_state().clone()
    models=fresh_pair('cpu')
    assert torch.equal(state,torch.get_rng_state())
    x,z=torch.randn(2,50,1024),torch.randn(2,50,1024)
    assert torch.equal(models[0](x,x),torch.zeros(2,50,7))
    assert torch.equal(models[1](x,z),torch.zeros(2,50,7))
    assert models[0].q.weight.data_ptr()!=models[1].q.weight.data_ptr()


def test_P_contains_nonzero_F_function_for_arbitrary_arrival():
    torch.manual_seed(51);f=Calibration();torch.nn.init.normal_(f.w2.weight,std=.02)
    p=copy.deepcopy(f)
    with torch.no_grad():
        p.w1.weight[:,256:512].copy_(f.w1.weight[:,256:512]+f.w1.weight[:,512:])
        p.w1.weight[:,512:].zero_()
    x,z=torch.randn(3,50,1024),torch.randn(3,50,1024)
    torch.testing.assert_close(f(x,x),p(x,z),rtol=2e-5,atol=2e-6)


def test_packed_credit_matches_two_independent_functions():
    torch.manual_seed(62);models=fresh_pair('cpu')
    for m in models:torch.nn.init.normal_(m.w2.weight,std=.02)
    other=copy.deepcopy(models);x,z=torch.randn(3,50,1024),torch.randn(3,50,1024)
    out=paired_forward(models,x,z)
    ref=torch.stack((other[0](x,x),other[1](x,z)))
    torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)
    out[:,:,:5].square().mean().backward();ref[:,:,:5].square().mean().backward()
    grads=torch.cat([p.grad.flatten() for p in models.parameters()])
    expected=torch.cat([p.grad.flatten() for p in other.parameters()])
    assert torch.isfinite(grads).all() and torch.linalg.vector_norm(grads-expected)/torch.linalg.vector_norm(expected)<2e-5


def test_events_exclude_internal_tasks_and_keep_each_task_weight():
    records=[dict(task=t,demo=d,interval_slots=list(range(4+(t%3)))) for t in FIT for d in range(16,20)]
    events=sample_events(records)
    assert events.shape==(500,160,3)
    assert set(events[:,:,0].flatten())==set(FIT)
    assert all(np.array_equal(row[:,0],np.repeat(FIT,8)) for row in events)
    assert np.array_equal(events,sample_events(records))
