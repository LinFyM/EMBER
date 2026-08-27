"""Bounded joint Program-query/native-key scalar compatibility."""

from __future__ import annotations

import math

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError


JOINT_SCALAR_INITIAL_SCALE = 0.03


class BoundedJointCompatibility(torch.nn.Module):
    """Score candidate content without emitting a native factor or lookup row.

    The residual dot product preserves the well-conditioned part of the prior
    scorer.  Additive attention supplies a non-separable content function for
    native directions that are absent from its linear key image.  Both paths
    end in one bounded scalar per query/candidate pair.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        if self.width <= 0:
            raise BankConditioningError("joint compatibility width is invalid")
        self.query_projection = torch.nn.Linear(self.width, self.width, bias=False)
        self.key_projection = torch.nn.Linear(self.width, self.width, bias=True)
        self.scalar = torch.nn.Linear(self.width, 1, bias=False)
        # Start as a small, nonzero residual around the established dot path.
        # Nonzero weights give query/key/scalar gradients on the first step;
        # the small amplitude avoids replacing the signed anchor before the
        # joint content function has learned useful compatibility.
        with torch.no_grad():
            self.scalar.weight.mul_(JOINT_SCALAR_INITIAL_SCALE)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Score `[rank,event,branch,width]` against event-indexed candidates."""

        if (
            query.ndim != 4
            or key.ndim < 3
            or query.shape[1] != key.shape[0]
            or query.shape[-1] != self.width
            or key.shape[-1] != self.width
        ):
            raise BankConditioningError("joint compatibility axes changed")
        events, ranks, branches = key.shape[0], query.shape[0], query.shape[2]
        candidate_shape = key.shape[1:-1]
        query_by_event = query.permute(1, 0, 2, 3)
        query_shape = (
            events,
            ranks,
            branches,
            *((1,) * len(candidate_shape)),
            self.width,
        )
        key_shape = (events, 1, 1, *candidate_shape, self.width)
        query_view = query_by_event.reshape(query_shape)
        key_view = key.reshape(key_shape)
        dot = (query_view * key_view).sum(-1) / math.sqrt(self.width)
        hidden = torch.tanh(
            self.query_projection(query_by_event).reshape(query_shape)
            + self.key_projection(key).reshape(key_shape)
        )
        joint = self.scalar(hidden).squeeze(-1)
        score = torch.tanh(dot + joint)
        order = (1, 0, 2, *range(3, score.ndim))
        return score.permute(order)
