"""Task-only Source native P/F calibration; retired after the fixed500 readback."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRAIN=(0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38)
INTERNAL=(0,12,20,32)
FIT=tuple(t for t in TRAIN if t not in INTERNAL)


def rms(x):
    value=x.float()
    return value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)


def attention(q,k,v):
    shape=q.shape
    def heads(x):return x.reshape(-1,50,4,64).transpose(1,2)
    out=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),dropout_p=0.)
    return out.transpose(1,2).reshape(*shape)


class Calibration(nn.Module):
    """The exact shared-C256 four-head residual function of contract§3."""
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(1024,256,bias=False)
        self.k=nn.Linear(1024,256,bias=False)
        self.v=nn.Linear(1024,256,bias=False)
        self.o=nn.Linear(256,256,bias=False)
        self.w1=nn.Linear(768,1024)
        self.w2=nn.Linear(1024,7)
        nn.init.zeros_(self.w2.weight);nn.init.zeros_(self.w2.bias)

    def forward(self,departure,context):
        x,z=rms(departure),rms(context)
        q=self.q(x)
        c0=self.o(attention(q,self.k(x),self.v(x)))
        c1=self.o(attention(q,self.k(z),self.v(z)))
        return self.w2(F.gelu(self.w1(torch.cat((q,c0,c1),-1))))


def fresh_pair(device):
    """Independent seed7 initialization without perturbing any source/probe RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        first=Calibration()
    return nn.ModuleList((first,copy.deepcopy(first))).to(device)


def paired_forward(models,departure,arrival):
    """Batch independent F/P linear/attention work; parameters and credit remain separate."""
    def linear(name,x):
        modules=[getattr(m,name) for m in models]
        weight=torch.stack([m.weight for m in modules])
        value=torch.einsum('abtd,awd->abtw',x,weight)
        return value+torch.stack([m.bias for m in modules])[:,None,None] if modules[0].bias is not None else value
    x=rms(departure)[None].expand(2,-1,-1,-1)
    z=rms(torch.stack((departure,arrival)))
    q=linear('q',x)
    c0=linear('o',attention(q,linear('k',x),linear('v',x)))
    c1=linear('o',attention(q,linear('k',z),linear('v',z)))
    return linear('w2',F.gelu(linear('w1',torch.cat((q,c0,c1),-1))))


def sample_events(records):
    """Uniform demo, then uniform complete interval;20 tasks×8, common500 stream."""
    rows={(r['task'],r['demo']):r for r in records}
    rng=np.random.default_rng(20261003)
    events=[]
    for _ in range(500):
        update=[]
        for task in FIT:
            for demo in rng.integers(16,20,size=8):
                row=rows[(task,int(demo))]
                slot=int(rng.choice(row['interval_slots']))
                update.append((task,int(demo),slot))
        events.append(update)
    return np.asarray(events,dtype=np.int64)


def optimization(models):
    opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-4) for m in models]
    scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda step:min((step+1)/10,1.)) for o in opts]
    return opts,scheds


def update(models,opts,scheds,departure,arrival,mu,target,mode):
    """All160 queries, full50 attention then real5×7 labels; each independentclip1."""
    for opt in opts:opt.zero_grad(set_to_none=True)
    if mode=='packed':
        outputs=paired_forward(models,departure,arrival)[:,:,:5]
        errors=(outputs.float()+mu[None]-target[None]).square().mean((2,3))
        losses=errors.reshape(2,20,8).mean((1,2))
        losses.sum().backward()
    elif mode=='sequential':
        losses=[]
        for i,model in enumerate(models):
            prediction=mu+model(departure,departure if i==0 else arrival)[:,:5].float()
            loss=(prediction-target).square().mean((1,2)).reshape(20,8).mean()
            loss.backward();losses.append(loss.detach())
        losses=torch.stack(losses)
    else:raise ValueError('unknown physical packing')
    norms=[float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)) for m in models]
    lrs=[o.param_groups[0]['lr'] for o in opts]
    for opt,sched in zip(opts,scheds,strict=True):opt.step();sched.step()
    return dict(loss_F=float(losses[0]),loss_P=float(losses[1]),unclipped_norm=norms,lr=lrs,
        intervals_per_arm=160,task_weight=1/20,physical_mode=mode)
