"""Targeted real gradient-boundary checks for the temporary §102 consumers."""
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from ember.operator_writer.model import TargetWrite
from ember.operator_writer import state_credit_diagnostic as sc
from ember.writer.function_credit import FlowSample


def test_original_recurrence_and_pre_output_content():
    torch.manual_seed(13)
    write = TargetWrite(12,9)
    nn.init.normal_(write.o.weight,std=.02)
    k = F.normalize(torch.randn(4,50,128),dim=-1).transpose(1,2)
    h = torch.randn(4,50,1024)
    m,z = sc.recurrence(write,k,h)
    ref = torch.zeros_like(m)
    nh = sc.rms(h)
    for t in range(3):
        u = F.gelu(write.p(k[t].T)+write.c(nh[t]))*write.d(nh[t+1]-nh[t])
        ref = ref + (write.o(u).T-ref@k[t])@k[t].T/50
    torch.testing.assert_close(m,ref,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(m,write.o(z.T).T,atol=2e-7,rtol=2e-5)
    (m.square().sum()+z.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in write.parameters())


def test_gamma_rng_zero_initialization_and_zero_value():
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    gamma = sc.CreditReader(targets=2,rank=3,hidden=4,width=8,heads=4)
    assert torch.equal(before,torch.get_rng_state())
    z,h = torch.randn(2,8,3),torch.randn(5,50,4)
    assert torch.count_nonzero(gamma(h,z))==0
    nn.init.normal_(gamma.o.weight)
    assert torch.count_nonzero(gamma(h,torch.zeros_like(z)))==0
    assert gamma.q.bias is gamma.k.bias is gamma.v.bias is gamma.o.bias is None


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.up = nn.Linear(3,4,bias=False)
        self.model.action_out_proj = nn.Linear(4,7,bias=False)
        self.model.denoise_step = lambda padding,cache,x,time: self.model.action_out_proj(self.model.up(x))


def compare_vjp(live):
    torch.manual_seed(9)
    policy = ToyPolicy().requires_grad_(False)
    phi = nn.Parameter(torch.randn(2))
    x,y = torch.randn(4,50,3),torch.randn(4,50,7)
    gamma = sc.CreditReader(targets=2,rank=3,hidden=4,width=8,heads=4)
    nn.init.normal_(gamma.o.weight,std=.1)
    original = {k:v.clone() for k,v in gamma.state_dict().items()}
    z_base = torch.randn(2,8,3)
    def compile():
        return {'model.up.weight':policy.model.up.weight+phi[0],
                'model.action_out_proj.weight':policy.model.action_out_proj.weight+phi[1]},\
                z_base+phi.sum()*torch.ones(2,8,3)
    def sample(p,b,**kwargs):
        return FlowSample((b['x'],),b['y'],7)
    def prepare(self,s):
        return (None,None,s.arguments[0],None)
    state,z = compile()
    runtime = SimpleNamespace(policy=policy,device=torch.device('cpu'))
    with patch.object(sc,'flow_sample',sample),patch.object(sc.ExecutionHidden,'prepare',prepare):
        credit = sc.coupled_credit(runtime,state,z,gamma,{'x':x,'y':y},seed=3,
            microbatch=2,live=live,backward=True)
    torch.autograd.backward([*state.values(),z],
        [*[credit['lora_cotangent'][k] for k in state],credit['Z_cotangent']])
    actual_phi = phi.grad.clone()
    actual_gamma = {k:p.grad.clone() for k,p in gamma.named_parameters()}
    phi.grad=None
    gamma.zero_grad(); gamma.load_state_dict(original)
    state,z = compile()
    h = F.linear(x,state['model.up.weight'])
    v = F.linear(h,state['model.action_out_proj.weight'])
    r = gamma(h if live else h.detach(),z)
    loss = ((v-y).square().mean()+(v.detach()+r-y).square().mean())/8
    loss.backward()
    torch.testing.assert_close(actual_phi,phi.grad,atol=1e-6,rtol=1e-5)
    for k,p in gamma.named_parameters():
        torch.testing.assert_close(actual_gamma[k],p.grad,atol=1e-6,rtol=1e-5)
    # The output head receives only the original FM credit, never auxiliary-query credit.
    expected_out = torch.einsum('bso,bsi->oi',2*(v.detach()-y)/(y.numel()*8),h.detach())
    torch.testing.assert_close(credit['lora_cotangent']['model.action_out_proj.weight'],expected_out)
    return actual_phi, float(loss.detach())


def test_live_stop_only_changes_hidden_credit_and_matches_direct_vjp():
    live,lf = compare_vjp(True)
    stop,sf = compare_vjp(False)
    assert abs(lf-sf)<1e-7
    assert abs(float(live[0]-stop[0]))>1e-5
    torch.testing.assert_close(live[1],stop[1])


def test_reader_hook_real_dimensions_only_and_is_removed():
    torch.manual_seed(4)
    policy = nn.Module(); policy.model=nn.Module(); policy.model.action_out_proj=nn.Linear(4,10,bias=False)
    gamma = sc.CreditReader(targets=2,rank=3,hidden=4,width=8,heads=4)
    nn.init.normal_(gamma.o.weight,std=.1)
    h,z = torch.randn(2,50,4),torch.randn(2,8,3)
    baseline=policy.model.action_out_proj(h)
    with sc.residual_output(policy,gamma,z):
        actual=policy.model.action_out_proj(h)
    torch.testing.assert_close(actual[...,:7],baseline[...,:7]+gamma(h,z))
    torch.testing.assert_close(actual[...,7:],baseline[...,7:])
    torch.testing.assert_close(policy.model.action_out_proj(h),baseline)
