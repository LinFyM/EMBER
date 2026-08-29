"""Natural Program (Pass A) for the ECP Native-Factor Compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ember.ecp.stage0 import ECPVideoEncoder, ECPVideoEncoderOutput


@dataclass(frozen=True)
class NaturalProgram:
    """The fixed deployment schema for one or more video conditions."""

    p_lang: torch.Tensor
    p_scene: torch.Tensor
    p_process: torch.Tensor
    rho: torch.Tensor
    tau: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class NaturalProgramPredictions:
    action_phases: torch.Tensor
    progress: torch.Tensor
    rising_logits: torch.Tensor
    contact_logits: torch.Tensor
    predicate_logits: torch.Tensor
    scene_predicate_logits: torch.Tensor


@dataclass(frozen=True)
class NaturalProgramOutput:
    program: NaturalProgram
    predictions: NaturalProgramPredictions | None
    local_scene: torch.Tensor
    local_process: torch.Tensor
    local_presence: torch.Tensor
    local_tau: torch.Tensor
    local_sigma: torch.Tensor
    probe_process: torch.Tensor
    probe_presence: torch.Tensor
    alignment: torch.Tensor
    canonical_assignment: torch.Tensor
    frame_mask: torch.Tensor
    video_condition_ids: torch.Tensor


@dataclass(frozen=True)
class FrozenProgramEvidence:
    """Deployment-visible frozen source/Stage0 evidence for Pass-A compilation."""

    language_embeddings: torch.Tensor
    language_mask: torch.Tensor
    patch_states: torch.Tensor
    frame_mask: torch.Tensor
    process: torch.Tensor
    uncertainty: torch.Tensor
    presence: torch.Tensor
    state_posterior: torch.Tensor
    frame_indices: torch.Tensor
    raw_frame_counts: torch.Tensor
    video_offsets: torch.Tensor
    video_set_offsets: torch.Tensor
    frame_condition_ids: torch.Tensor


class OwnerLanguageReader(torch.nn.Module):
    """Let every deployed LoRA owner read the exact frozen language tokens."""

    def __init__(self, *, owners: int, token_width: int, width: int) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(owners, width))
        self.key = torch.nn.Linear(token_width, width, bias=False)
        self.value = torch.nn.Linear(token_width, width, bias=False)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width),
            torch.nn.GELU(),
            torch.nn.LayerNorm(width),
        )
        torch.nn.init.normal_(self.queries, std=width**-0.5)

    def forward(
        self, language_tokens: torch.Tensor, language_mask: torch.Tensor
    ) -> torch.Tensor:
        key = self.key(language_tokens)
        value = self.value(language_tokens)
        logits = torch.einsum("jd,cld->cjl", self.queries, key) / math.sqrt(
            key.shape[-1]
        )
        logits = logits.masked_fill(
            ~language_mask[:, None], torch.finfo(logits.dtype).min
        )
        attended = torch.einsum("cjl,cld->cjd", logits.softmax(-1), value)
        query = self.queries[None].expand(language_tokens.shape[0], -1, -1)
        return self.output(torch.cat((query, attended), dim=-1))


class OwnerSceneReader(torch.nn.Module):
    """Read first/final patches and their relation independently per owner."""

    def __init__(self, *, owners: int, width: int) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(owners, width))
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(4 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        torch.nn.init.normal_(self.queries, std=width**-0.5)

    def _read(self, patches: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        key = self.key(patches)
        value = self.value(patches)
        logits = torch.einsum("vjd,vpd->vjp", query, key) / math.sqrt(
            key.shape[-1]
        )
        return torch.einsum("vjp,vpd->vjd", logits.softmax(-1), value)

    def forward(
        self,
        patch_states: torch.Tensor,
        frame_mask: torch.Tensor,
        p_lang: torch.Tensor,
        video_condition_ids: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.arange(patch_states.shape[0], device=patch_states.device)
        final_indices = frame_mask.sum(1).clamp_min(1) - 1
        first_patches = patch_states[:, 0]
        final_patches = patch_states[rows, final_indices]
        language = p_lang.index_select(0, video_condition_ids)
        query = self.queries[None] + language
        first = self._read(first_patches, query)
        final = self._read(final_patches, query)
        return self.output(torch.cat((language, first, final, final - first), dim=-1))


class MonotonicCanonicalAligner(torch.nn.Module):
    """Soft forward-only alignment from local ordered slots to canonical slots."""

    def __init__(self, *, event_slots: int, width: int) -> None:
        super().__init__()
        self.event_slots = event_slots
        self.canonical_queries = torch.nn.Parameter(
            torch.empty(event_slots, width)
        )
        self.event_key = torch.nn.Linear(width, width, bias=False)
        self.time_key = torch.nn.Linear(2, width, bias=False)
        self.log_time_scale = torch.nn.Parameter(torch.tensor(0.0))
        torch.nn.init.normal_(self.canonical_queries, std=width**-0.5)

    def _posterior(self, emission: torch.Tensor) -> torch.Tensor:
        batch, local_slots, canonical_slots = emission.shape
        if local_slots != self.event_slots or canonical_slots != self.event_slots:
            raise ValueError("Natural Program event alignment changed capacity")
        source = torch.arange(canonical_slots, device=emission.device)[:, None]
        destination = torch.arange(canonical_slots, device=emission.device)[None]
        allowed = destination >= source
        gap = (destination - source).to(emission.dtype)
        negative = torch.finfo(emission.dtype).min
        transition = (-gap).masked_fill(~allowed, negative).log_softmax(-1)
        # Local and canonical sequences have the same fixed maximum capacity.
        # Anchor their boundaries while keeping every intermediate stay/skip
        # path available.  Without the terminal constraints the learned
        # content score can route every local event through one canonical slot
        # even though the native observer still exposes a full ordered path.
        start = emission.new_full((canonical_slots,), negative)
        start[0] = 0.0

        alpha = emission.new_empty(batch, local_slots, canonical_slots)
        alpha[:, 0] = emission[:, 0] + start
        for local in range(1, local_slots):
            alpha[:, local] = emission[:, local] + torch.logsumexp(
                alpha[:, local - 1, :, None] + transition[None], dim=1
            )

        beta = emission.new_full(
            (batch, local_slots, canonical_slots), negative
        )
        beta[:, -1, -1] = 0.0
        for local in range(local_slots - 2, -1, -1):
            beta[:, local] = torch.logsumexp(
                transition[None]
                + emission[:, local + 1, None]
                + beta[:, local + 1, None],
                dim=-1,
            )
        return (alpha + beta).softmax(-1)

    def forward(
        self, process: torch.Tensor, presence: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        event = process.mean(2)
        key = self.event_key(event) + self.time_key(tau)
        content = torch.einsum(
            "ved,cd->vec", key, self.canonical_queries
        ) / math.sqrt(key.shape[-1])
        anchors = torch.linspace(
            0.0,
            1.0,
            self.event_slots,
            dtype=tau.dtype,
            device=tau.device,
        )
        time_penalty = (tau[..., :1] - anchors).square()
        scale = self.log_time_scale.exp().clamp(max=100.0)
        emission = presence[..., None] * (content - scale * time_penalty)
        # The DP posterior is [video, local, canonical].  Consumers use
        # [video, canonical, local] so every row pools local events.
        return self._posterior(emission).transpose(1, 2)


class TemporalProgramDecoder(torch.nn.Module):
    """Training-only heads that test whether full dynamics survive the schema."""

    def __init__(
        self,
        *,
        width: int,
        owners: int,
        action_phases: int,
        predicate_slots: int,
    ) -> None:
        super().__init__()
        self.action_phases = action_phases
        self.predicate_slots = predicate_slots
        self.owner_queries = torch.nn.Parameter(torch.empty(owners, width))
        torch.nn.init.uniform_(
            self.owner_queries[:1], -width**-0.5, width**-0.5
        )
        with torch.no_grad():
            self.owner_queries[1:].copy_(self.owner_queries[:1])
        self.scene_owner_score = torch.nn.Linear(width, 1, bias=False)
        self.query_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
        )
        self.action_head = torch.nn.Linear(width, action_phases * 7)
        self.progress_head = torch.nn.Linear(width, 1)
        self.rising_head = torch.nn.Linear(width, 1)
        self.contact_head = torch.nn.Linear(width, 1)
        self.predicate_head = torch.nn.Linear(width, predicate_slots)
        self.scene_predicate_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * predicate_slots),
        )

    def forward(
        self, program: NaturalProgram, query_times: torch.Tensor
    ) -> NaturalProgramPredictions:
        # The temporal heads are a G2 mechanism test, so they may only read the
        # event-bearing Program fields.  Adding P_lang/P_scene to every event
        # gives the decoder a task/endpoint code that can fit cross-episode
        # action priors while ignoring the ordered video process entirely.
        owner_context = program.p_process
        if owner_context.shape[-2] != self.owner_queries.shape[0]:
            raise ValueError("Natural Program owner axis changed")
        owner_weights = torch.einsum(
            "cejd,jd->cej", torch.tanh(owner_context), self.owner_queries
        )
        event_tokens = torch.einsum(
            "cej,cejd->ced", owner_weights.softmax(-1), owner_context
        )
        center = program.tau[..., 0]
        width = program.tau[..., 1].clamp_min(0.04)
        distance = (query_times[:, :, None] - center[:, None]).square()
        event_weights = program.rho[:, None] * torch.exp(
            -0.5 * distance / width[:, None].square()
        )
        event_weights = event_weights / event_weights.sum(-1, keepdim=True).clamp_min(
            1e-6
        )
        states = self.query_projection(
            torch.einsum("cqe,ced->cqd", event_weights, event_tokens)
        )
        scene_context = program.p_scene + program.p_lang
        scene_weights = self.scene_owner_score(
            torch.tanh(scene_context)
        ).squeeze(-1).softmax(-1)
        scene_state = torch.einsum("cj,cjd->cd", scene_weights, scene_context)
        action = self.action_head(states).reshape(
            *states.shape[:-1], self.action_phases, 7
        )
        return NaturalProgramPredictions(
            action_phases=action,
            progress=self.progress_head(states).squeeze(-1).sigmoid(),
            rising_logits=self.rising_head(states).squeeze(-1),
            contact_logits=self.contact_head(states).squeeze(-1),
            predicate_logits=self.predicate_head(states),
            scene_predicate_logits=self.scene_predicate_head(scene_state).reshape(
                scene_state.shape[0], 2, self.predicate_slots
            ),
        )


class NaturalProgramModel(torch.nn.Module):
    """Two-probe Pass A followed by monotonic, uniform Dynamic-K aggregation."""

    def __init__(
        self,
        encoder: ECPVideoEncoder,
        *,
        prefix_width: int = 2048,
        width: int = 128,
        owners: int = 38,
        event_slots: int = 8,
        action_phases: int = 10,
        predicate_slots: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.width = width
        self.owners = owners
        self.event_slots = event_slots
        self.language_reader = OwnerLanguageReader(
            owners=owners, token_width=prefix_width, width=width
        )
        self.scene_reader = OwnerSceneReader(owners=owners, width=width)
        self.process_fusion = torch.nn.Sequential(
            # Preserve a static-free process path.  The frozen native process
            # is already task-grounded by exact language; uncertainty is the
            # only additional local event field fused here.  P_lang/P_scene
            # remain available in their own Program fields and the scene head.
            torch.nn.Linear(2 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.aligner = MonotonicCanonicalAligner(
            event_slots=event_slots, width=width
        )
        self.decoder = TemporalProgramDecoder(
            width=width,
            owners=owners,
            action_phases=action_phases,
            predicate_slots=predicate_slots,
        )

    @staticmethod
    def _pad_video_time(
        values: tuple[torch.Tensor, ...], *, time_dim: int, fill: float = 0.0
    ) -> torch.Tensor:
        maximum = max(value.shape[time_dim] for value in values)
        shape = list(values[0].shape)
        shape[0] = len(values)
        shape[time_dim] = maximum
        output = values[0].new_full(shape, fill)
        for row, value in enumerate(values):
            selection = [slice(None)] * output.ndim
            selection[0] = row
            selection[time_dim] = slice(0, value.shape[time_dim])
            source = value[0]
            output[tuple(selection)] = source
        return output

    def _encode_videos_independently(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        suffix_noise: torch.Tensor,
    ) -> ECPVideoEncoderOutput:
        boundaries = video_offsets.detach().cpu().tolist()
        rows = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            rows.append(
                self.encoder(
                    policy=policy,
                    frames=frames[start:stop],
                    video_offsets=torch.tensor(
                        [0, stop - start],
                        dtype=torch.long,
                        device=frames.device,
                    ),
                    frame_condition_ids=frame_condition_ids[start:stop],
                    language_tokens=language_tokens,
                    language_mask=language_mask,
                    suffix_noise=suffix_noise,
                    action_meta_lora=None,
                    install_action_meta_lora=False,
                )
            )
        values = tuple(rows)
        return ECPVideoEncoderOutput(
            process=torch.cat(tuple(row.process for row in values)),
            presence=torch.cat(tuple(row.presence for row in values)),
            uncertainty=torch.cat(tuple(row.uncertainty for row in values)),
            assignment=self._pad_video_time(
                tuple(row.assignment for row in values), time_dim=2
            ),
            state_posterior=self._pad_video_time(
                tuple(row.state_posterior for row in values), time_dim=1
            ),
            confidence=self._pad_video_time(
                tuple(row.confidence for row in values), time_dim=1, fill=-20.0
            ),
            frame_mask=self._pad_video_time(
                tuple(row.frame_mask for row in values), time_dim=1
            ),
            program_summary=torch.cat(
                tuple(row.program_summary for row in values)
            ),
            frame_owner_evidence=self._pad_video_time(
                tuple(row.frame_owner_evidence for row in values), time_dim=1
            ),
            patch_states=self._pad_video_time(
                tuple(row.patch_states for row in values), time_dim=1
            ),
            language_summary=torch.cat(
                tuple(row.language_summary for row in values)
            ),
            scene_transition=torch.cat(
                tuple(row.scene_transition for row in values)
            ),
        )

    @staticmethod
    def _ordinary_evidence(value: torch.Tensor) -> torch.Tensor:
        """Turn an inference tensor into an ordinary detached cache tensor."""

        return value.detach().clone()

    def encode_frozen_evidence(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        raw_frame_counts: torch.Tensor,
        video_offsets: torch.Tensor,
        video_set_offsets: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> FrozenProgramEvidence:
        """Run only frozen source/Stage0 work and return cacheable evidence."""

        with torch.inference_mode():
            language_embeddings = self.encoder.embed_language_conditions(
                policy, language_tokens
            )
            common = {
                "policy": policy,
                "frames": frames,
                "video_offsets": video_offsets,
                "frame_condition_ids": frame_condition_ids,
                "language_tokens": language_tokens,
                "language_mask": language_mask,
            }
            positive = self._encode_videos_independently(
                **common, suffix_noise=self.encoder.fixed_suffix_noise
            )
            negative = self._encode_videos_independently(
                **common, suffix_noise=-self.encoder.fixed_suffix_noise
            )
            stacked = {
                "process": torch.stack((positive.process, negative.process)),
                "uncertainty": torch.stack(
                    (positive.uncertainty, negative.uncertainty)
                ),
                "presence": torch.stack((positive.presence, negative.presence)),
                "state_posterior": torch.stack(
                    (positive.state_posterior, negative.state_posterior)
                ),
            }
            patch_states = positive.patch_states
            frame_mask = positive.frame_mask
        ordinary = self._ordinary_evidence
        return FrozenProgramEvidence(
            language_embeddings=ordinary(language_embeddings),
            language_mask=ordinary(language_mask),
            patch_states=ordinary(patch_states),
            frame_mask=ordinary(frame_mask),
            process=ordinary(stacked["process"]),
            uncertainty=ordinary(stacked["uncertainty"]),
            presence=ordinary(stacked["presence"]),
            state_posterior=ordinary(stacked["state_posterior"]),
            frame_indices=ordinary(frame_indices),
            raw_frame_counts=ordinary(raw_frame_counts),
            video_offsets=ordinary(video_offsets),
            video_set_offsets=ordinary(video_set_offsets),
            frame_condition_ids=ordinary(frame_condition_ids),
        )

    @staticmethod
    def _padded_positions(
        frame_indices: torch.Tensor,
        raw_frame_counts: torch.Tensor,
        video_offsets: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = frame_mask.new_zeros(frame_mask.shape, dtype=torch.float32)
        boundaries = video_offsets.detach().cpu().tolist()
        for video, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            denominator = max(int(raw_frame_counts[video]) - 1, 1)
            positions[video, : stop - start] = (
                frame_indices[start:stop].float() / denominator
            )
        return positions

    @staticmethod
    def _temporal_moments(
        posterior: torch.Tensor,
        positions: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask[None, :, :, None].to(posterior.dtype)
        weights = posterior * valid
        mass = weights.sum(2).clamp_min(1e-6)
        center = torch.einsum("pvte,vt->pve", weights, positions) / mass
        second = torch.einsum("pvte,vt->pve", weights, positions.square()) / mass
        spread = (second - center.square()).clamp_min(1e-4).sqrt()
        return torch.stack((center, spread), dim=-1).mean(0)

    def _local_program(
        self,
        evidence: FrozenProgramEvidence,
        *,
        p_lang: torch.Tensor,
        video_condition_ids: torch.Tensor,
        frame_indices: torch.Tensor,
        raw_frame_counts: torch.Tensor,
        video_offsets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        frame_mask = evidence.frame_mask
        p_scene = self.scene_reader(
            evidence.patch_states,
            frame_mask,
            p_lang,
            video_condition_ids,
        )
        process = evidence.process
        uncertainty = evidence.uncertainty
        presence = evidence.presence
        probe_program = []
        for probe in range(2):
            probe_program.append(
                self.process_fusion(
                    torch.cat(
                        (
                            process[probe],
                            uncertainty[probe],
                        ),
                        dim=-1,
                    )
                )
            )
        probe_programs = torch.stack(probe_program)
        local_process = probe_programs.mean(0)
        local_sigma = (
            torch.stack(
                (
                    uncertainty.square().mean(0),
                    (probe_programs - local_process[None]).square().mean(0),
                )
            ).sum(0)
        ).clamp_min(1e-6).sqrt()
        local_presence = presence.mean(0)
        positions = self._padded_positions(
            frame_indices, raw_frame_counts, video_offsets, frame_mask
        )
        posterior = evidence.state_posterior
        local_tau = self._temporal_moments(posterior, positions, frame_mask)
        return (
            p_scene,
            local_process,
            local_presence,
            local_tau,
            local_sigma,
            probe_programs,
            presence,
        )

    @staticmethod
    def _align_values(
        alignment: torch.Tensor,
        process: torch.Tensor,
        presence: torch.Tensor,
        tau: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = alignment * presence[:, None]
        mass = weights.sum(-1).clamp_min(1e-6)
        aligned_process = torch.einsum(
            "vce,vejd->vcjd", weights, process
        ) / mass[:, :, None, None]
        aligned_tau = torch.einsum("vce,veu->vcu", weights, tau) / mass[:, :, None]
        second = torch.einsum(
            "vce,vejd->vcjd", weights, sigma.square() + process.square()
        ) / mass[:, :, None, None]
        aligned_sigma = (second - aligned_process.square()).clamp_min(1e-6).sqrt()
        aligned_presence = 1.0 - torch.exp(
            torch.log1p(-(alignment * presence[:, None]).clamp(max=1.0 - 1e-6)).sum(
                -1
            )
        )
        return aligned_process, aligned_presence, aligned_tau, aligned_sigma

    def _aggregate(
        self,
        *,
        p_lang: torch.Tensor,
        local_scene: torch.Tensor,
        local_process: torch.Tensor,
        local_presence: torch.Tensor,
        local_tau: torch.Tensor,
        local_sigma: torch.Tensor,
        video_set_offsets: torch.Tensor,
    ) -> tuple[NaturalProgram, torch.Tensor]:
        boundaries = video_set_offsets.detach().cpu().tolist()
        if len(boundaries) != p_lang.shape[0] + 1:
            raise ValueError("Natural Program video sets do not match conditions")
        scene_rows = []
        process_rows = []
        presence_rows = []
        tau_rows = []
        sigma_rows = []
        alignments = []
        identity = torch.eye(
            self.event_slots,
            dtype=local_process.dtype,
            device=local_process.device,
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            count = stop - start
            if count not in (1, 2, 4):
                raise ValueError("Natural Program requires K in {1,2,4}")
            scene = local_scene[start:stop]
            process = local_process[start:stop]
            presence = local_presence[start:stop]
            tau = local_tau[start:stop]
            sigma = local_sigma[start:stop]
            if count == 1:
                alignment = identity[None]
                aligned_process = process
                aligned_presence = presence
                aligned_tau = tau
                aligned_sigma = sigma
            else:
                alignment = self.aligner(process, presence, tau)
                (
                    aligned_process,
                    aligned_presence,
                    aligned_tau,
                    aligned_sigma,
                ) = self._align_values(
                    alignment, process, presence, tau, sigma
                )
            # G2 has no learned video reliability: beta_k is exactly 1/K.
            # The explicit branch keeps every K=1 field bitwise identical to
            # its independently encoded video program.
            if count == 1:
                scene_rows.append(scene[0])
                process_rows.append(aligned_process[0])
                presence_rows.append(aligned_presence[0])
                tau_rows.append(aligned_tau[0])
                sigma_rows.append(aligned_sigma[0])
            else:
                # Uniform beta_k is a probability measure, so accumulate the
                # small K set in FP32.  This keeps permutation invariance from
                # depending on BF16 reduction order without changing K=1.
                mean_process = aligned_process.float().mean(0)
                scene_rows.append(scene.float().mean(0))
                process_rows.append(mean_process)
                presence_rows.append(aligned_presence.float().mean(0))
                tau_rows.append(aligned_tau.float().mean(0))
                sigma_rows.append(
                    (
                        aligned_sigma.float().square()
                        + (aligned_process.float() - mean_process[None]).square()
                    ).mean(0).clamp_min(1e-6).sqrt()
                )
            alignments.append(alignment)
        return (
            NaturalProgram(
                p_lang=p_lang,
                p_scene=torch.stack(scene_rows),
                p_process=torch.stack(process_rows),
                rho=torch.stack(presence_rows),
                tau=torch.stack(tau_rows),
                sigma=torch.stack(sigma_rows),
            ),
            torch.cat(alignments),
        )

    def compile_program(
        self,
        evidence: FrozenProgramEvidence,
        *,
        query_times: torch.Tensor,
        decode_predictions: bool = True,
    ) -> NaturalProgramOutput:
        """Compile cached frozen evidence through the differentiable Program."""

        p_lang = self.language_reader(
            evidence.language_embeddings, evidence.language_mask
        )
        video_condition_ids = evidence.frame_condition_ids.index_select(
            0, evidence.video_offsets[:-1].to(evidence.frame_condition_ids.device)
        )
        (
            local_scene,
            local_process,
            local_presence,
            local_tau,
            local_sigma,
            probe_process,
            probe_presence,
        ) = self._local_program(
            evidence,
            p_lang=p_lang,
            video_condition_ids=video_condition_ids,
            frame_indices=evidence.frame_indices,
            raw_frame_counts=evidence.raw_frame_counts,
            video_offsets=evidence.video_offsets,
        )
        program, alignment = self._aggregate(
            p_lang=p_lang,
            local_scene=local_scene,
            local_process=local_process,
            local_presence=local_presence,
            local_tau=local_tau,
            local_sigma=local_sigma,
            video_set_offsets=evidence.video_set_offsets,
        )
        local_assignment = evidence.state_posterior.mean(0)
        canonical_assignment = torch.einsum(
            "vte,vce->vtc", local_assignment, alignment
        )
        canonical_assignment = canonical_assignment / canonical_assignment.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        canonical_assignment = canonical_assignment * evidence.frame_mask[..., None]
        return NaturalProgramOutput(
            program=program,
            predictions=(
                self.decoder(program, query_times) if decode_predictions else None
            ),
            local_scene=local_scene,
            local_process=local_process,
            local_presence=local_presence,
            local_tau=local_tau,
            local_sigma=local_sigma,
            probe_process=probe_process,
            probe_presence=probe_presence,
            alignment=alignment,
            canonical_assignment=canonical_assignment,
            frame_mask=evidence.frame_mask,
            video_condition_ids=video_condition_ids,
        )

    def forward(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        raw_frame_counts: torch.Tensor,
        video_offsets: torch.Tensor,
        video_set_offsets: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> NaturalProgramOutput:
        evidence = self.encode_frozen_evidence(
            policy=policy,
            frames=frames,
            frame_indices=frame_indices,
            raw_frame_counts=raw_frame_counts,
            video_offsets=video_offsets,
            video_set_offsets=video_set_offsets,
            frame_condition_ids=frame_condition_ids,
            language_tokens=language_tokens,
            language_mask=language_mask,
        )
        return self.compile_program(evidence, query_times=query_times)
