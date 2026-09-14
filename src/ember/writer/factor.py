"""Fixed source pullback and shared two-sided native coordinate transforms."""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


class SharedSourceOutlet(torch.nn.Module):
    """Task-independent L B0 A0 R, with rank-sized residual transforms.

    The transforms act in native input/output coordinates, so a rotation of
    the internal PCA rank basis cancels between B0 and A0. No dense identity
    or task-conditioned parameter path is constructed.
    """

    def __init__(self, contract):
        super().__init__()
        self.contract = contract
        self.left_u, self.left_v = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.right_u, self.right_v = torch.nn.ParameterList(), torch.nn.ParameterList()
        for target in contract.targets:
            for width, up, down in ((target.out_features, self.left_u, self.left_v),
                                    (target.in_features, self.right_u, self.right_v)):
                up.append(torch.nn.Parameter(torch.zeros(width, contract.rank)))
                down.append(torch.nn.Parameter(torch.randn(contract.rank, width) * width ** -.5))

    def forward(self, bare_state):
        state = {}
        for target, left_u, left_v, right_u, right_v in zip(
                self.contract.targets, self.left_u, self.left_v, self.right_u, self.right_v, strict=True):
            a, b = bare_state[target.name + LORA_A_SUFFIX], bare_state[target.name + LORA_B_SUFFIX]
            # Factors and shared transforms retain FP32 under the Writer's
            # outer autocast. Native source and actual main FM are unchanged.
            with torch.autocast(a.device.type, enabled=False):
                state[target.name + LORA_A_SUFFIX] = a + (a @ right_u) @ right_v
                state[target.name + LORA_B_SUFFIX] = b + left_u @ (left_v @ b)
        return state


@torch.no_grad()
def native_input_basis(gram: torch.Tensor, rank: int) -> torch.Tensor:
    """Top right singular vectors of uncentered native X, from its full Gram.

    Eigenvector signs and bases inside tied eigenspaces are not selected by
    outcomes. B and A use the same basis, so sign changes cancel in B A.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not 0 < rank <= len(gram):
        raise ValueError("source input projection requires a square full-video Gram and legal rank")
    if not torch.isfinite(gram).all():
        raise ValueError("source input Gram is nonfinite")
    with torch.autocast(gram.device.type, enabled=False):
        _, vectors = torch.linalg.eigh(gram.float())
    return vectors[:, -rank:].flip(-1).T.contiguous().cpu()


class _SourcePullback(torch.autograd.Function):
    """A fixed linear map; retain no source graph or q values between its passes."""

    @staticmethod
    def forward(ctx, q, coordinates):
        ctx.coordinates, ctx.q_device, ctx.q_dtype = coordinates, q.device, q.dtype
        ctx.set_materialize_grads(False)
        return tuple(value.to(q.device) for value in coordinates.reader.pullback(coordinates, q))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        # Reverse-over-reverse differentiates the native VJP with respect to
        # its dummy output cotangent, not with respect to any source weight.
        gradient = ctx.coordinates.reader.adjoint(ctx.coordinates, gradients)
        return gradient.to(device=ctx.q_device, dtype=ctx.q_dtype), None


def compile_source_lora(coordinates, q: torch.Tensor) -> dict[str, torch.Tensor]:
    """One complete alpha=rank adapter, with frozen A and differentiable B(q)."""
    if (q.shape != coordinates.predictions.shape or not q.is_floating_point()
            or not torch.isfinite(q).all()):
        raise ValueError("source pullback q must cover every real frame and all 50x7 action coordinates")
    values = _SourcePullback.apply(q, coordinates)
    state = {}
    for target, value in zip(coordinates.reader.contract.targets, values, strict=True):
        state[target.name + LORA_A_SUFFIX] = coordinates.bases[target.name].to(q.device)
        state[target.name + LORA_B_SUFFIX] = value
    return state
