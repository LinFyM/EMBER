"""Fixed source ten-flow Jacobian in all 38 complete weight-operator coordinates.

The two custom linear maps are mutual adjoints. Their backwards use the same
first derivative at source, never a source Hessian or a factor tangent space.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ember.writer.practice.execution import NativeVelocity, action_chunk, native_actions
from ember.writer.runtime import autocast
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


class DenseExecution:
    """Temporary full additive operators; the physical source remains frozen."""
    def __init__(self, runtime):
        self.runtime, self.active, self.handles = runtime, None, []
        self.targets = tuple(runtime.lora.targets)
        for ordinal, target in enumerate(self.targets):
            module = runtime.policy.get_submodule(target.name)
            self.handles.append(module.register_forward_hook(self._hook(ordinal)))

    def _hook(self, ordinal):
        def add(_module, inputs, output):
            if self.active is None:
                return output
            with torch.autocast(output.device.type, enabled=False):
                delta = F.linear(inputs[0].float(), self.active[ordinal])
            return (output + delta).to(output.dtype)
        return add

    @contextmanager
    def activate(self, operators):
        if self.active is not None or self.runtime.execution._active_state is not None:
            raise ValueError('dense source linearization overlaps task LoRA execution')
        if len(operators) != 38:
            raise ValueError('the native Jacobian requires all 38 complete operators')
        self.active = operators
        try:
            yield
        finally:
            self.active = None

    def zeros(self, *, gradients=False):
        return tuple(torch.zeros(t.out_features, t.in_features, device=self.runtime.device,
                                 dtype=torch.float32, requires_grad=gradients) for t in self.targets)

    def close(self):
        if self.active is not None:
            raise RuntimeError('cannot close an active dense operator execution')
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class NativeJacobian:
    """One immutable real-H support, noise and source prefix shared by J and Jᵀ."""
    def __init__(self, runtime, dense, batch, noise, *, microbatch=4):
        self.runtime, self.dense, self.microbatch = runtime, dense, int(microbatch)
        self.noise = noise.detach().to(runtime.device, dtype=torch.float32)
        if self.noise.ndim != 3 or self.noise.shape[1:] != (50,32) or not len(self.noise):
            raise ValueError('source support lost real native 50x32 noise')
        self.count, self.parts = len(noise), []
        self.calls = dict(response=0, VJP=0, JVP=0, velocity_calls=0, velocity_rows=0)
        with torch.no_grad(), autocast(runtime.device):
            for first in range(0,self.count,self.microbatch):
                last = min(first+self.microbatch,self.count)
                part = {k:v[first:last] for k,v in batch.items()}
                velocity = NativeVelocity(runtime.policy,part,capture_phi=False)
                self.parts.append((first,last,velocity))

    def _actions(self, velocity, noise, operators, *, checkpointed):
        def denoise(z, tau):
            def call(value, *matrices):
                self.calls['velocity_calls'] += 1
                self.calls['velocity_rows'] += len(value)
                with self.dense.activate(matrices):
                    return velocity(value,tau)
            return (checkpoint(call,z,*operators,use_reentrant=False,preserve_rng_state=False)
                    if checkpointed else call(z,*operators))
        with autocast(self.runtime.device):
            output,_ = action_chunk(velocity,noise,capture_hidden=False,denoise=denoise)
        return output[:,:5].float()

    @torch.no_grad()
    def response(self, operators=None):
        self.calls['response'] += 1
        operators = self.dense.zeros() if operators is None else operators
        return torch.cat([self._actions(v,self.noise[a:b],operators,checkpointed=False)
                          for a,b,v in self.parts])

    @torch.no_grad()
    def controller_response(self,state):
        """The actual exported complete factor execution, on this same support."""
        self.calls['actual_LoRA_response']=self.calls.get('actual_LoRA_response',0)+1
        with autocast(self.runtime.device):
            return torch.cat([native_actions(self.runtime,[state],{},self.noise[a:b],
                batch_indices=torch.zeros(b-a,device=self.runtime.device,dtype=torch.long),
                checkpointed=False,prepared_velocity=v).float() for a,b,v in self.parts])

    def vjp(self, pressure):
        if pressure.shape != (self.count,5,7):
            raise ValueError('source pullback needs one cotangent per actual first5x7 response')
        self.calls['VJP'] += 1
        total = None
        for first,last,velocity in self.parts:
            with torch.enable_grad():
                operators = self.dense.zeros(gradients=True)
                output = self._actions(velocity,self.noise[first:last],operators,checkpointed=True)
                values = torch.autograd.grad(output,operators,pressure[first:last].detach().float())
            if total is None:
                total = [x.detach().float() for x in values]
            else:
                for dest, value in zip(total,values,strict=True):
                    dest.add_(value.detach().float())
        return tuple(total)

    @torch.no_grad()
    def jvp(self, operators):
        self.calls['JVP'] += 1
        outputs = []
        # Forward AD computes the real first derivative through all ten calls.
        # Eager source attention supports dual tensors; no finite differences.
        for first,last,velocity in self.parts:
            # Autograd.Function.forward/backward disable forward AD by default.
            # These fixed linear-map implementations explicitly own the JVP.
            with torch.autograd.forward_ad._set_fwd_grad_enabled(True),torch.autograd.forward_ad.dual_level():
                dual = tuple(torch.autograd.forward_ad.make_dual(z,d.detach().float())
                             for z,d in zip(self.dense.zeros(),operators,strict=True))
                output = self._actions(velocity,self.noise[first:last],dual,checkpointed=False)
                tangent = torch.autograd.forward_ad.unpack_dual(output).tangent
                if tangent is None:
                    raise RuntimeError('native JVP lost the complete operator tangent')
                outputs.append(tangent.detach().float())
        return torch.cat(outputs)

    def close(self):
        self.parts.clear()


class _Pullback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pressure, jacobian):
        ctx.jacobian = jacobian
        return tuple(v/(35*jacobian.count) for v in jacobian.vjp(pressure))

    @staticmethod
    def backward(ctx, *operators):
        return ctx.jacobian.jvp(operators)/(35*ctx.jacobian.count), None


class _Pushforward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, jacobian, *operators):
        ctx.jacobian = jacobian
        return jacobian.jvp(operators)

    @staticmethod
    def backward(ctx, pressure):
        return (None,*ctx.jacobian.vjp(pressure))


def public_probes(lora, device):
    return tuple(torch.randn(t.in_features,8,generator=torch.Generator().manual_seed(2026101700+i),
                             dtype=torch.float32).div(t.in_features**.5).to(device)
                 for i,t in enumerate(lora.targets))


def rank8_range(G, omega):
    """Differentiable normalized QR plus exactly two registered power iterations."""
    norm = torch.linalg.vector_norm(G)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('nonfinite native full-operator pullback')
    if norm.item() == 0:
        return G.new_zeros(G.shape[0],8), G.new_zeros(8,G.shape[1]), G*0
    Z = G/norm
    Q,R = torch.linalg.qr(Z@omega,mode='reduced')
    if not bool(torch.isfinite(Q).all()) or bool((R.diagonal()==0).any()):
        raise FloatingPointError('actual QR range is singular/nonfinite; no jitter or fallback')
    for _ in range(2):
        Q,R = torch.linalg.qr(Z@(Z.T@Q),mode='reduced')
        if not bool(torch.isfinite(Q).all()) or bool((R.diagonal()==0).any()):
            raise FloatingPointError('actual QR power range is singular/nonfinite')
    C = Q.T@G
    return Q,C,Q@C


def write_operators(jacobian, pressure, probes, lora):
    """Full G/r mutual credit, live QR/alpha, pure-gauge balanced complete A/B."""
    if pressure.dtype != torch.float32:
        raise ValueError('pressure/QR/functional calibration must aggregate in FP32')
    G = _Pullback.apply(pressure,jacobian)
    with torch.autocast(pressure.device.type,enabled=False):
        ranges = [rank8_range(g,o) for g,o in zip(G,probes,strict=True)]
        D = tuple(x[2] for x in ranges)
        response = _Pushforward.apply(jacobian,*D)
        denominator = response.square().sum()
        numerator = (pressure*response).sum()
        if not bool(torch.isfinite(denominator)) or not bool(torch.isfinite(numerator)):
            raise FloatingPointError('nonfinite same-Jacobian functional calibration')
        zero = denominator.item()==0
        alpha = pressure.new_zeros(()) if zero else numerator/denominator
        if not zero and (not bool(torch.isfinite(alpha)) or alpha.item()<=0):
            raise FloatingPointError('nonpositive native calibration; no epsilon/abs/clip')
        state = {}
        for t,(Q,C,_),g in zip(lora.targets,ranges,G,strict=True):
            if zero or torch.linalg.vector_norm(Q).item()==0:
                A,B = C*0,Q*0
            else:
                gauge = (torch.linalg.vector_norm(C)/(alpha*torch.linalg.vector_norm(Q))).sqrt().detach()
                A,B = C/gauge,-alpha*gauge*Q
            state[t.name+LORA_A_SUFFIX] = A
            state[t.name+LORA_B_SUFFIX] = B
        energy = sum(d.square().sum() for d in D)
        jacobian.calibration=dict(alpha=float(alpha.detach()),pressure=pressure.detach(),r=response.detach())
        diagnostic = dict(alpha=float(alpha.detach()),pressure_norm=float(pressure.detach().norm()),
            G_norm=float(torch.stack([g.detach().norm() for g in G]).norm()),
            rank8_energy=float(energy.detach()),response_norm=float(denominator.detach().sqrt()),
            adjoint_left=float((numerator/(35*jacobian.count)).detach()),
            adjoint_right=float(energy.detach()),zero_direction=zero)
        return state,diagnostic
