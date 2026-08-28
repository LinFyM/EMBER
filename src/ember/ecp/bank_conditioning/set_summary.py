"""Low-dimensional set summaries that score exact native X/Y candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.native_factors import G1_RESIDUAL_RANK, rms_normalize

if TYPE_CHECKING:
    from ember.ecp.bank_conditioning.native_bank_runtime import NativeCandidateBank


@dataclass(frozen=True)
class SetSummaryStatistics:
    """Per-event unit-measure first and diagonal second centered moments."""

    value: torch.Tensor
    event_mass: torch.Tensor


class StreamingSetMoments:
    """Accumulate low-dimensional set moments without retaining candidates."""

    def __init__(self, *, events: int, width: int, reference: torch.Tensor) -> None:
        if min(events, width) <= 0:
            raise BankConditioningError("invalid set-summary moment topology")
        self.events = int(events)
        self.width = int(width)
        self.mass = reference.new_zeros(events, dtype=torch.float32)
        self.first = reference.new_zeros(events, width, dtype=torch.float32)
        self.second = reference.new_zeros(events, width, dtype=torch.float32)
        self.candidate_count = 0

    def add(self, features: torch.Tensor, event_mass: torch.Tensor) -> None:
        if (
            features.ndim < 2
            or features.shape[-1] != self.width
            or event_mass.shape != (self.events, *features.shape[:-1])
        ):
            raise BankConditioningError("set-summary candidate axes changed")
        flat = features.float().reshape(-1, self.width)
        mass = event_mass.detach().float().reshape(self.events, -1)
        if (
            torch.any(mass < 0)
            or not bool(torch.isfinite(mass).all())
            or not bool(torch.isfinite(flat).all())
        ):
            raise BankConditioningError("set-summary stream is invalid")
        self.mass = self.mass + mass.sum(-1)
        self.first = self.first + torch.einsum("en,nw->ew", mass, flat)
        self.second = self.second + torch.einsum(
            "en,nw->ew", mass, flat.square()
        )
        self.candidate_count += int(flat.shape[0])

    def finalize(self) -> SetSummaryStatistics:
        if self.candidate_count <= 0 or not torch.any(self.mass > 0):
            raise BankConditioningError("set-summary stream is empty")
        denominator = self.mass[:, None].clamp_min(1e-12)
        mean = self.first / denominator
        variance = (self.second / denominator - mean.square()).clamp_min(0)
        active = self.mass > 0
        value = torch.cat((mean, variance), dim=-1)
        value = torch.where(active[:, None], value, torch.zeros_like(value))
        return SetSummaryStatistics(value=value, event_mass=self.mass)


class TaskLocalSelectionCode(torch.nn.Module):
    """Capacity-only free rank/event code shared by every video of one task."""

    def __init__(self, *, events: int, width: int) -> None:
        super().__init__()
        if min(events, width) <= 0:
            raise BankConditioningError("invalid task-local selection code")
        self.code = torch.nn.Parameter(torch.empty(G1_RESIDUAL_RANK, events, width))
        self.event_logits = torch.nn.Parameter(
            torch.zeros(G1_RESIDUAL_RANK, events)
        )
        torch.nn.init.normal_(self.code, std=width**-0.5)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.code, self.event_logits.softmax(-1)


class SetConditionedScalarEnergy(torch.nn.Module):
    """Score each current candidate from content, set context, and a small code."""

    def __init__(
        self,
        *,
        feature_width: int,
        event_slots: int,
        global_events: bool,
        hidden_width: int,
        logit_bound: float,
    ) -> None:
        super().__init__()
        if (
            min(feature_width, event_slots, hidden_width) <= 0
            or logit_bound <= 0
        ):
            raise BankConditioningError("invalid set-conditioned energy topology")
        self.feature_width = int(feature_width)
        self.event_slots = int(event_slots)
        self.global_events = bool(global_events)
        self.logit_bound = float(logit_bound)
        self.summary_features = self._feature_network(feature_width)
        summary_width = 2 * feature_width
        if self.global_events:
            self.summary_context = torch.nn.Sequential(
                torch.nn.LayerNorm(event_slots * summary_width),
                torch.nn.Linear(event_slots * summary_width, hidden_width),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, event_slots * feature_width),
            )
        else:
            self.summary_context = torch.nn.Sequential(
                torch.nn.LayerNorm(summary_width),
                torch.nn.Linear(summary_width, hidden_width),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, feature_width),
            )
        self.code_context = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_width),
            torch.nn.Linear(feature_width, feature_width),
        )
        self.candidate_basis = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_width),
            torch.nn.Linear(feature_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, 2 * feature_width),
        )
        self.condition_coefficients = torch.nn.Sequential(
            torch.nn.LayerNorm(3 * feature_width),
            torch.nn.Linear(3 * feature_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, 2 * feature_width + 2),
        )
        self._reset_energy()

    @staticmethod
    def _feature_network(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )

    def _reset_energy(self) -> None:
        candidate = self.candidate_basis[-1]
        condition = self.condition_coefficients[-1]
        with torch.no_grad():
            torch.nn.init.normal_(candidate.weight[: self.feature_width], std=0.02)
            candidate.weight[self.feature_width :].copy_(
                candidate.weight[: self.feature_width]
            )
            candidate.bias.zero_()
            torch.nn.init.normal_(
                condition.weight[: self.feature_width], std=0.02
            )
            condition.weight[
                self.feature_width : 2 * self.feature_width
            ].copy_(-condition.weight[: self.feature_width])
            condition.weight[2 * self.feature_width :].zero_()
            condition.bias.zero_()

    def summarize(
        self, keys: torch.Tensor, event_mass: torch.Tensor
    ) -> SetSummaryStatistics:
        encoded = self.summary_features(keys.float())
        accumulator = StreamingSetMoments(
            events=self.event_slots,
            width=self.feature_width,
            reference=encoded,
        )
        accumulator.add(encoded, event_mass)
        return accumulator.finalize()

    def _context(self, statistics: SetSummaryStatistics) -> torch.Tensor:
        if statistics.value.shape != (
            self.event_slots,
            2 * self.feature_width,
        ):
            raise BankConditioningError("set-summary statistics changed shape")
        if self.global_events:
            return self.summary_context(statistics.value.reshape(1, -1)).reshape(
                self.event_slots, self.feature_width
            )
        return self.summary_context(statistics.value)

    def score(
        self,
        keys: torch.Tensor,
        code: torch.Tensor,
        statistics: SetSummaryStatistics,
        *,
        topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidate_shape = keys.shape[:-1]
        if (
            keys.shape[-1] != self.feature_width
            or code.shape
            != (G1_RESIDUAL_RANK, self.event_slots, self.feature_width)
        ):
            raise BankConditioningError("set-conditioned score axes changed")
        context = self._context(statistics)
        code_context = self.code_context(code.float())
        if topology is not None:
            if topology.shape != (self.feature_width,):
                raise BankConditioningError("set-conditioned topology changed")
            code_context = code_context + topology
        event_context = context[None].expand(G1_RESIDUAL_RANK, -1, -1)
        condition = self.condition_coefficients(
            torch.cat(
                (code_context, event_context, code_context * event_context), dim=-1
            )
        )
        coefficients = condition[..., : 2 * self.feature_width].reshape(
            G1_RESIDUAL_RANK,
            self.event_slots,
            2,
            self.feature_width,
        )
        branch_bias = condition[..., 2 * self.feature_width :]
        basis = self.candidate_basis(keys.float()).reshape(
            -1, 2, self.feature_width
        )
        logits = torch.einsum("nbw,rebw->renb", basis, coefficients)
        logits = logits / self.feature_width**0.5 + branch_bias[:, :, None]
        logits = self.logit_bound * torch.tanh(logits)
        return logits.reshape(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *candidate_shape,
            2,
        )


class StreamingSetSignedPool:
    """Online softmax of explicit scalar branches followed by event mixing."""

    def __init__(self, *, ranks: int, events: int, width: int, reference: torch.Tensor):
        if min(ranks, events, width) <= 0:
            raise BankConditioningError("invalid set-signed pool topology")
        self.ranks = int(ranks)
        self.events = int(events)
        self.width = int(width)
        shape = (ranks, events, 2)
        self.maximum = reference.new_full(shape, -torch.inf, dtype=torch.float32)
        self.normalizer = reference.new_zeros(shape, dtype=torch.float32)
        self.weighted_sum = reference.new_zeros(
            *shape, width, dtype=torch.float32
        )
        self.candidate_count = 0

    def add(
        self,
        values: torch.Tensor,
        event_mass: torch.Tensor,
        branch_logits: torch.Tensor,
    ) -> None:
        candidate_shape = values.shape[:-1]
        expected = (self.ranks, self.events, *candidate_shape, 2)
        if (
            values.ndim < 2
            or values.shape[-1] != self.width
            or event_mass.shape != (self.events, *candidate_shape)
            or branch_logits.shape != expected
        ):
            raise BankConditioningError("set-signed pool candidate axes changed")
        value = values.detach().float().reshape(-1, self.width)
        mass = event_mass.detach().float().reshape(self.events, -1)
        score = branch_logits.float().reshape(
            self.ranks, self.events, -1, 2
        ).permute(0, 1, 3, 2)
        if torch.any(mass < 0) or not bool(torch.isfinite(score).all()):
            raise BankConditioningError("set-signed pool stream is invalid")
        log_mass = torch.where(
            mass > 0,
            mass.clamp_min(1e-30).log(),
            torch.full_like(mass, -torch.inf),
        )
        logits = score + log_mass[None, :, None]
        chunk_maximum = logits.amax(-1)
        maximum = torch.maximum(self.maximum, chunk_maximum).detach()
        finite = torch.isfinite(maximum)
        old_scale = torch.where(
            torch.isfinite(self.maximum),
            torch.exp(self.maximum - torch.where(finite, maximum, self.maximum)),
            torch.zeros_like(maximum),
        )
        shift = torch.where(finite, maximum, torch.zeros_like(maximum))
        weights = torch.where(
            torch.isfinite(logits),
            torch.exp(logits - shift[..., None]),
            torch.zeros_like(logits),
        )
        self.weighted_sum = self.weighted_sum * old_scale[..., None] + torch.einsum(
            "rebn,nd->rebd", weights, value
        )
        self.normalizer = self.normalizer * old_scale + weights.sum(-1)
        self.maximum = maximum
        self.candidate_count += int(value.shape[0])

    def signed_factor(self, event_weights: torch.Tensor) -> torch.Tensor:
        if (
            self.candidate_count <= 0
            or event_weights.shape != (self.ranks, self.events)
        ):
            raise BankConditioningError("set-signed pool is empty")
        active = self.normalizer.amin(-1) > 0
        if not torch.all(active.any(-1)):
            raise BankConditioningError("set-signed pool has no active event")
        branch = self.weighted_sum / self.normalizer.clamp_min(1e-30)[..., None]
        signed = branch[..., 0, :] - branch[..., 1, :]
        weights = event_weights.float() * active
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
        return torch.einsum("re,red->rd", weights, signed)


def materialized_set_signed_pool(
    values: torch.Tensor,
    event_mass: torch.Tensor,
    branch_logits: torch.Tensor,
    event_weights: torch.Tensor,
) -> torch.Tensor:
    accumulator = StreamingSetSignedPool(
        ranks=branch_logits.shape[0],
        events=branch_logits.shape[1],
        width=values.shape[-1],
        reference=branch_logits,
    )
    accumulator.add(values, event_mass, branch_logits)
    return accumulator.signed_factor(event_weights)


class SetSummaryFactorSelector(torch.nn.Module):
    """Produce one rank-four residual by exact input/output candidate pooling."""

    def __init__(
        self,
        *,
        feature_width: int,
        event_slots: int,
        output_groups: int,
        global_events: bool,
        hidden_width: int,
        logit_bound: float,
    ) -> None:
        super().__init__()
        if output_groups <= 0:
            raise BankConditioningError("set-summary selector has no output group")
        arguments = dict(
            feature_width=feature_width,
            event_slots=event_slots,
            global_events=global_events,
            hidden_width=hidden_width,
            logit_bound=logit_bound,
        )
        self.input_energy = SetConditionedScalarEnergy(**arguments)
        self.output_energy = SetConditionedScalarEnergy(**arguments)
        self.output_group_embedding = torch.nn.Parameter(
            torch.empty(output_groups, feature_width)
        )
        torch.nn.init.normal_(self.output_group_embedding, std=feature_width**-0.5)

    @staticmethod
    def _factor(
        energy: SetConditionedScalarEnergy,
        bank: NativeCandidateBank,
        code: torch.Tensor,
        event_weights: torch.Tensor,
        topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        statistics = energy.summarize(bank.content_keys, bank.event_mass)
        logits = energy.score(
            bank.content_keys, code, statistics, topology=topology
        )
        return materialized_set_signed_pool(
            bank.values, bank.event_mass, logits, event_weights
        )

    def forward(
        self,
        *,
        input_bank: NativeCandidateBank,
        output_banks: Sequence[NativeCandidateBank],
        code: torch.Tensor,
        event_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(output_banks) != self.output_group_embedding.shape[0]:
            raise BankConditioningError("set-summary output groups changed")
        input_factor = self._factor(
            self.input_energy, input_bank, code, event_weights
        )
        output_blocks = tuple(
            self._factor(
                self.output_energy,
                bank,
                code,
                event_weights,
                self.output_group_embedding[group],
            )
            for group, bank in enumerate(output_banks)
        )
        return rms_normalize(input_factor), rms_normalize(
            torch.cat(output_blocks, dim=-1)
        )
