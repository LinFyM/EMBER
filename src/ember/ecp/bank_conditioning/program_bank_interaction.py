"""Candidate-level Program--bank corrections for exact native signed pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)


@dataclass(frozen=True)
class ProgramBankContext:
    """Deployment-visible local Program evidence owned by one video bank."""

    canonical_assignment: torch.Tensor
    frame_positions: torch.Tensor
    local_scene: torch.Tensor
    local_process: torch.Tensor
    local_presence: torch.Tensor
    local_tau: torch.Tensor
    local_sigma: torch.Tensor

    def frame_slice(self, start: int, stop: int) -> ProgramBankContext:
        if not 0 <= start < stop <= self.frame_positions.shape[0]:
            raise BankConditioningError("Program-bank frame slice changed")
        return ProgramBankContext(
            canonical_assignment=self.canonical_assignment[start:stop],
            frame_positions=self.frame_positions[start:stop],
            local_scene=self.local_scene,
            local_process=self.local_process,
            local_presence=self.local_presence,
            local_tau=self.local_tau,
            local_sigma=self.local_sigma,
        )


@dataclass(frozen=True)
class ProgramBankInteractionState:
    """Shared Program queries paired with one bank's local context."""

    context: ProgramBankContext
    rank_event: torch.Tensor
    event_weights: torch.Tensor
    input_event_queries: tuple[torch.Tensor, ...]
    output_event_queries: tuple[torch.Tensor, ...]


class ProgramBankInteractionScorer(torch.nn.Module):
    """Produce bounded per-candidate corrections, never factors or routes."""

    _metadata_width = 8
    _interaction_width = 32
    _scalar_feature_width = 5 + _metadata_width
    _correction_feature_width = _scalar_feature_width + _interaction_width

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        semantic_width: int = 32,
        hidden_width: int = 64,
        correction_bound: float = 0.1,
        replay_score_rms: float,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.semantic_width = int(semantic_width)
        self.hidden_width = int(hidden_width)
        self.correction_bound = float(correction_bound)
        self.replay_score_rms = float(replay_score_rms)
        if (
            not self.owners
            or min(
                self.program_width,
                self.event_slots,
                self.semantic_width,
                self.hidden_width,
            )
            <= 0
            or self.semantic_width != self._interaction_width
            or not 0.0 < self.correction_bound <= 1.0
            or not math.isfinite(self.replay_score_rms)
            or self.replay_score_rms <= 0.0
        ):
            raise BankConditioningError("invalid Program-bank interaction topology")
        families = tuple(TargetFamily)
        self._native_widths = self._family_side_widths(self.owners)
        self.native_query_projection = torch.nn.ModuleDict(
            {
                side: torch.nn.ModuleDict(
                    {
                        family.value: torch.nn.Linear(
                            self._native_widths[side][family.value],
                            self._interaction_width,
                            bias=False,
                        )
                        for family in families
                        if family.value in self._native_widths[side]
                    }
                )
                for side in ("input", "output")
            }
        )
        self.native_key_projection = torch.nn.ModuleDict(
            {
                side: torch.nn.ModuleDict(
                    {
                        family.value: torch.nn.Linear(
                            self._native_widths[side][family.value],
                            self._interaction_width,
                            bias=False,
                        )
                        for family in families
                        if family.value in self._native_widths[side]
                    }
                )
                for side in ("input", "output")
            }
        )
        for side in ("input", "output"):
            for family in self.native_query_projection[side]:
                query = self.native_query_projection[side][family]
                candidate = self.native_key_projection[side][family]
                torch.nn.init.orthogonal_(query.weight)
                with torch.no_grad():
                    candidate.weight.copy_(query.weight)
        self.program_query = torch.nn.ModuleDict(
            {
                family.value: torch.nn.Linear(
                    self.program_width, self.semantic_width, bias=False
                )
                for family in families
            }
        )
        local_width = 3 * self.program_width + 3
        self.local_key = torch.nn.ModuleDict(
            {
                family.value: torch.nn.Sequential(
                    torch.nn.LayerNorm(local_width),
                    torch.nn.Linear(local_width, 2 * self.semantic_width),
                    torch.nn.GELU(),
                    torch.nn.Linear(2 * self.semantic_width, self.semantic_width),
                    torch.nn.LayerNorm(self.semantic_width),
                )
                for family in families
            }
        )
        self.metadata_key = torch.nn.ModuleDict(
            {
                family.value: torch.nn.Linear(
                    self._metadata_width, self.semantic_width, bias=False
                )
                for family in families
            }
        )
        self.correction = torch.nn.ModuleDict(
            {
                family.value: torch.nn.Sequential(
                    torch.nn.LayerNorm(self._correction_feature_width),
                    torch.nn.Linear(
                        self._correction_feature_width, self.hidden_width
                    ),
                    torch.nn.GELU(),
                    torch.nn.Linear(self.hidden_width, 1),
                )
                for family in families
            }
        )
        for family in families:
            torch.nn.init.zeros_(self.correction[family.value][-1].weight)
            torch.nn.init.zeros_(self.correction[family.value][-1].bias)

    @staticmethod
    def _family_side_widths(
        owners: Sequence[TargetOwner],
    ) -> dict[str, dict[str, int]]:
        widths: dict[str, dict[str, int]] = {"input": {}, "output": {}}
        for owner in owners:
            current = {
                "input": owner.in_features,
                "output": owner.out_features // native_output_group_count(owner),
            }
            for side, width in current.items():
                family = owner.family.value
                previous = widths[side].setdefault(family, width)
                if width <= 0 or previous != width:
                    raise BankConditioningError(
                        f"Program-bank {side} native width changed within {family}"
                    )
        return widths

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        return value / value.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()

    def _base_score_feature(
        self, base_query: torch.Tensor, centered: torch.Tensor
    ) -> torch.Tensor:
        """Expose the detached B1 base score without changing candidate measure."""

        candidate_axes = centered.shape[:-1]
        if base_query.shape != (G1_RESIDUAL_RANK, centered.shape[-1]):
            raise BankConditioningError("Program-bank base-query axes changed")
        score = torch.einsum(
            "rd,...d->r...", base_query.detach().float(), centered.detach().float()
        ) / self.replay_score_rms
        return score.reshape(
            G1_RESIDUAL_RANK, 1, *candidate_axes, 1
        ).expand(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *candidate_axes,
            1,
        )

    def _vector_interaction(
        self,
        *,
        family: str,
        side: str,
        native_query: torch.Tensor,
        candidate_unit: torch.Tensor,
        semantic_query: torch.Tensor,
        local_key: torch.Tensor,
        metadata_key: torch.Tensor,
        candidate_axes: tuple[int, ...],
    ) -> torch.Tensor:
        native_query_vector = self._rms_normalize(
            self.native_query_projection[side][family](native_query)
        )
        native_candidate_vector = self._rms_normalize(
            self.native_key_projection[side][family](candidate_unit)
        )
        semantic_key = self._rms_normalize(
            local_key.reshape(
                self.event_slots,
                *((1,) * len(candidate_axes)),
                self._interaction_width,
            )
            + metadata_key.reshape(1, *candidate_axes, self._interaction_width)
        )
        query_interaction = native_query_vector * semantic_query
        candidate_interaction = (
            native_candidate_vector.reshape(1, *candidate_axes, self._interaction_width)
            * semantic_key
        )
        return query_interaction.reshape(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *((1,) * len(candidate_axes)),
            self._interaction_width,
        ) * candidate_interaction.reshape(
            1,
            self.event_slots,
            *candidate_axes,
            self._interaction_width,
        )

    def _validate_context(self, context: ProgramBankContext) -> int:
        frames = int(context.frame_positions.shape[0])
        targets = len(self.owners)
        if (
            frames <= 0
            or context.frame_positions.shape != (frames,)
            or context.canonical_assignment.shape != (frames, self.event_slots)
            or context.local_scene.shape != (targets, self.program_width)
            or context.local_process.shape
            != (self.event_slots, targets, self.program_width)
            or context.local_presence.shape != (self.event_slots,)
            or context.local_tau.shape != (self.event_slots, 2)
            or context.local_sigma.shape
            != (self.event_slots, targets, self.program_width)
        ):
            raise BankConditioningError("Program-bank local context changed")
        return frames

    def _metadata(
        self, context: ProgramBankContext, *, output: bool, like: torch.Tensor
    ) -> torch.Tensor:
        frames = context.frame_positions.to(like).clamp(0.0, 1.0)
        probes = torch.linspace(-1.0, 1.0, G1_PROBE_COUNT, device=like.device)
        horizons = torch.linspace(-1.0, 1.0, ACTION_HORIZON, device=like.device)
        if output:
            types = functional.one_hot(
                torch.arange(len(OUTPUT_BANK_TYPES), device=like.device),
                num_classes=len(OUTPUT_BANK_TYPES),
            ).to(dtype=like.dtype)
            shape = (
                frames.shape[0],
                G1_PROBE_COUNT,
                ACTION_HORIZON,
                len(OUTPUT_BANK_TYPES),
            )
            return torch.cat(
                (
                    frames[:, None, None, None, None].expand(*shape, 1),
                    probes[None, :, None, None, None].expand(*shape, 1),
                    horizons[None, None, :, None, None].expand(*shape, 1),
                    types[None, None, None].expand(*shape, len(OUTPUT_BANK_TYPES)),
                    torch.ones(*shape, 1, device=like.device, dtype=like.dtype),
                ),
                dim=-1,
            )
        shape = (frames.shape[0], G1_PROBE_COUNT, ACTION_HORIZON)
        return torch.cat(
            (
                frames[:, None, None, None].expand(*shape, 1),
                probes[None, :, None, None].expand(*shape, 1),
                horizons[None, None, :, None].expand(*shape, 1),
                torch.zeros(
                    *shape,
                    len(OUTPUT_BANK_TYPES) + 1,
                    device=like.device,
                    dtype=like.dtype,
                ),
            ),
            dim=-1,
        )

    def _corrections(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        event_weights: torch.Tensor,
        base_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        context: ProgramBankContext,
        output: bool,
    ) -> torch.Tensor:
        frames = self._validate_context(context)
        owner = self.owners[target]
        candidate_axes = (
            (frames, G1_PROBE_COUNT, ACTION_HORIZON, len(OUTPUT_BANK_TYPES))
            if output
            else (frames, G1_PROBE_COUNT, ACTION_HORIZON)
        )
        width = values.shape[-1]
        side = "output" if output else "input"
        family = owner.family.value
        if (
            program_event_state.shape
            != (G1_RESIDUAL_RANK, self.event_slots, self.program_width)
            or native_event_query.shape
            != (G1_RESIDUAL_RANK, self.event_slots, width)
            or event_weights.shape != (G1_RESIDUAL_RANK, self.event_slots)
            or base_query.shape != (G1_RESIDUAL_RANK, width)
            or values.shape != (*candidate_axes, width)
            or native_mean.shape != (width,)
            or self._native_widths[side].get(family) != width
        ):
            raise BankConditioningError("Program-bank candidate axes changed")

        centered = values.detach().float() - native_mean.detach().float()
        candidate_unit = self._rms_normalize(centered)
        native_query = self._rms_normalize(native_event_query.float())
        native_alignment = torch.einsum(
            "red,...d->re...", native_query, candidate_unit
        ) / math.sqrt(width)

        local = torch.cat(
            (
                context.local_scene[target].float()[None].expand(
                    self.event_slots, -1
                ),
                context.local_process[:, target].float(),
                context.local_sigma[:, target].float(),
                context.local_presence.float()[:, None],
                context.local_tau.float(),
            ),
            dim=-1,
        )
        semantic_query = self._rms_normalize(
            self.program_query[family](program_event_state.float())
        )
        local_key = self._rms_normalize(self.local_key[family](local))
        metadata = self._metadata(context, output=output, like=values.float())
        metadata_key = self.metadata_key[family](metadata.float())
        semantic_alignment = (
            torch.einsum("res,es->re", semantic_query, local_key).reshape(
                G1_RESIDUAL_RANK,
                self.event_slots,
                *((1,) * len(candidate_axes)),
            )
            + torch.einsum("res,...s->re...", semantic_query, metadata_key)
        ) / math.sqrt(self.semantic_width)
        vector_interaction = self._vector_interaction(
            family=family,
            side=side,
            native_query=native_query,
            candidate_unit=candidate_unit,
            semantic_query=semantic_query,
            local_key=local_key,
            metadata_key=metadata_key,
            candidate_axes=candidate_axes,
        )

        log_norm = centered.square().mean(-1).clamp_min(1e-12).sqrt().log()
        expanded_metadata = metadata.reshape(
            1, 1, *candidate_axes, self._metadata_width
        ).expand(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *candidate_axes,
            self._metadata_width,
        )
        features = torch.cat(
            (
                native_alignment[..., None],
                semantic_alignment[..., None],
                (native_alignment * semantic_alignment)[..., None],
                self._base_score_feature(base_query, centered),
                log_norm.reshape(1, 1, *candidate_axes, 1).expand(
                    G1_RESIDUAL_RANK,
                    self.event_slots,
                    *candidate_axes,
                    1,
                ),
                expanded_metadata,
                vector_interaction,
            ),
            dim=-1,
        )
        event_correction = self.correction_bound * torch.tanh(
            self.correction[family](features).squeeze(-1)
        )
        assignment = context.canonical_assignment.float().T.reshape(
            1,
            self.event_slots,
            frames,
            *((1,) * (len(candidate_axes) - 1)),
        )
        weights = event_weights.float().reshape(
            G1_RESIDUAL_RANK,
            self.event_slots,
            *((1,) * len(candidate_axes)),
        )
        correction = (event_correction * assignment * weights).sum(1)
        # Subtracting the global measure mean is a per-rank softmax gauge:
        # it cancels exactly inside each branch. Keeping this equivalent
        # representative avoids a third full-video streaming pass.
        return torch.stack((correction, -correction), dim=1)

    def input_logit_corrections(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        event_weights: torch.Tensor,
        base_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        context: ProgramBankContext,
    ) -> torch.Tensor:
        arguments = {
            "target": target,
            "program_event_state": program_event_state,
            "native_event_query": native_event_query,
            "event_weights": event_weights,
            "base_query": base_query,
            "values": values,
            "native_mean": native_mean,
            "context": context,
            "output": False,
        }
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._corrections, use_reentrant=False, **arguments)
        return self._corrections(**arguments)

    def output_logit_corrections(
        self,
        *,
        target: int,
        program_event_state: torch.Tensor,
        native_event_query: torch.Tensor,
        event_weights: torch.Tensor,
        base_query: torch.Tensor,
        values: torch.Tensor,
        native_mean: torch.Tensor,
        context: ProgramBankContext,
    ) -> torch.Tensor:
        arguments = {
            "target": target,
            "program_event_state": program_event_state,
            "native_event_query": native_event_query,
            "event_weights": event_weights,
            "base_query": base_query,
            "values": values,
            "native_mean": native_mean,
            "context": context,
            "output": True,
        }
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._corrections, use_reentrant=False, **arguments)
        return self._corrections(**arguments)
