"""Privileged task-local Programs for the fixed-compiler reachability oracle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.program import ECPProgram
from ember.ecp.stage1_data import pack_stage1_videos


class TaskLocalFreeProgram(torch.nn.Module):
    """Optimize only the q_pi-writable Program fields around one frozen anchor."""

    def __init__(
        self,
        program: ECPProgram,
        *,
        process_delta_scale: float,
        uncertainty_log_scale_bound: float,
    ) -> None:
        super().__init__()
        if (
            program.language.shape[0] != 1
            or program.scene.shape[0] != 1
            or program.process.shape[0] != 1
            or program.presence.shape[0] != 1
            or program.uncertainty.shape != program.process.shape
            or process_delta_scale <= 0.0
            or uncertainty_log_scale_bound <= 0.0
        ):
            raise ValueError("invalid task-local free Program initialization")
        self.process_delta_scale = float(process_delta_scale)
        self.uncertainty_log_scale_bound = float(uncertainty_log_scale_bound)
        self.register_buffer("language", program.language[0].detach().float().clone())
        self.register_buffer("scene", program.scene[0].detach().float().clone())
        self.register_buffer(
            "base_process", program.process[0].detach().float().clone()
        )
        self.register_buffer("presence", program.presence[0].detach().float().clone())
        self.register_buffer(
            "base_uncertainty",
            program.uncertainty[0].detach().float().clamp_min(1e-4).clone(),
        )
        self.process_delta = torch.nn.Parameter(torch.zeros_like(self.base_process))
        self.uncertainty_log_scale = torch.nn.Parameter(
            torch.zeros_like(self.base_uncertainty)
        )

    def forward(self) -> ECPProgram:
        process_correction = self.process_delta_scale * torch.tanh(self.process_delta)
        uncertainty_scale = torch.exp(
            self.uncertainty_log_scale_bound * torch.tanh(self.uncertainty_log_scale)
        )
        return ECPProgram(
            language=self.language[None],
            scene=self.scene[None],
            process=(self.base_process + process_correction)[None],
            presence=self.presence[None],
            uncertainty=(self.base_uncertainty * uncertainty_scale)[None],
        )

    def base_program(self) -> ECPProgram:
        return ECPProgram(
            language=self.language[None],
            scene=self.scene[None],
            process=self.base_process[None],
            presence=self.presence[None],
            uncertainty=self.base_uncertainty[None],
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        correction = self.process_delta_scale * torch.tanh(self.process_delta)
        scale = torch.exp(
            self.uncertainty_log_scale_bound * torch.tanh(self.uncertainty_log_scale)
        )
        return {
            "process_delta_relative": correction.float().square().sum().sqrt()
            / self.base_process.float().square().sum().sqrt().clamp_min(1e-8),
            "uncertainty_scale_mean": scale.float().mean(),
            "uncertainty_scale_min": scale.float().min(),
            "uncertainty_scale_max": scale.float().max(),
        }


class TaskLocalFreeProgramTable(torch.nn.Module):
    """A fit-task-only diagnostic table that can never serve deployment."""

    def __init__(
        self,
        programs: Mapping[int, ECPProgram],
        *,
        process_delta_scale: float,
        uncertainty_log_scale_bound: float,
    ) -> None:
        super().__init__()
        ordinals = tuple(sorted(int(value) for value in programs))
        if not ordinals:
            raise ValueError("free Program table has no tasks")
        self._ordinals = ordinals
        self.register_buffer("task_ordinals", torch.tensor(ordinals, dtype=torch.int64))
        self.rows = torch.nn.ModuleList(
            [
                TaskLocalFreeProgram(
                    programs[ordinal],
                    process_delta_scale=process_delta_scale,
                    uncertainty_log_scale_bound=uncertainty_log_scale_bound,
                )
                for ordinal in ordinals
            ]
        )
        self._index = {ordinal: index for index, ordinal in enumerate(ordinals)}

    @property
    def ordinals(self) -> tuple[int, ...]:
        return self._ordinals

    def row(self, ordinal: int) -> TaskLocalFreeProgram:
        try:
            return self.rows[self._index[int(ordinal)]]
        except KeyError as error:
            raise ValueError("task is outside the free Program oracle") from error

    def forward(self, ordinal: int) -> ECPProgram:
        return self.row(ordinal)()

    def parameters_for_ordinals(
        self, ordinals: Sequence[int]
    ) -> tuple[torch.nn.Parameter, ...]:
        selected = []
        for ordinal in sorted({int(value) for value in ordinals}):
            selected.extend(self.row(ordinal).parameters())
        return tuple(selected)

    def freeze_inactive_gradients(self, active_ordinals: Sequence[int]) -> None:
        active = {int(value) for value in active_ordinals}
        for ordinal, row in zip(self.ordinals, self.rows, strict=True):
            if ordinal not in active:
                for parameter in row.parameters():
                    parameter.grad = None


def free_program_ordinals(config: Mapping[str, Any], *, mode: str) -> tuple[int, ...]:
    fit = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    return fit[:1] if mode == "profile" else fit


def initialize_task_local_free_programs(
    *,
    mode: str,
    config: Mapping[str, Any],
    context: Any,
    inputs: Any,
    policy: torch.nn.Module,
    observer: Any,
    model: torch.nn.Module,
    support_bank: Any,
    language_tokens: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> TaskLocalFreeProgramTable:
    """Capture one frozen q_pi anchor per fit task, distributed without repeats."""

    ordinals = free_program_ordinals(config, mode=mode)
    owners = int(config["model"]["target_owners"])
    events = int(config["model"]["event_slots"])
    width = int(config["model"]["program_width"])
    count = len(ordinals)
    tensors = {
        "language": torch.zeros(count, owners, width, device=context.device),
        "scene": torch.zeros(count, owners, width, device=context.device),
        "process": torch.zeros(count, events, owners, width, device=context.device),
        "presence": torch.zeros(count, events, device=context.device),
        "uncertainty": torch.zeros(count, events, owners, width, device=context.device),
    }
    coverage = torch.zeros(count, device=context.device)
    visit = int(config["free_program_oracle"]["initialization_video_visit"])
    expert = policy.model.paligemma_with_expert.gemma_expert.model
    try:
        for index, ordinal in enumerate(ordinals):
            if index % context.world_size != context.rank:
                continue
            packed = pack_stage1_videos(
                store=inputs.video_store,
                ordinal=ordinal,
                visit=visit,
                seed=int(config["data"]["pair_seed"]),
                k=int(config["data"]["visible_videos_per_visit"]),
                device=context.device,
            )
            tokens, mask = language_tokens[ordinal]
            with torch.no_grad(), observer.action_meta.installed(expert):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = observer.model.encoder(
                        policy=policy,
                        frames=packed.frames,
                        video_offsets=packed.video_offsets,
                        frame_condition_ids=packed.frame_condition_ids,
                        language_tokens=tokens,
                        language_mask=mask,
                    )
                    evidence = inputs.evidence_bank.evidence(
                        ordinal, support_bank.task(ordinal)
                    )
                    anchors = model.visible_program(
                        encoded, packed.video_group_ids, group_count=1
                    )
                    program = model.policy_teacher(anchors, evidence).program
            for name, target in tensors.items():
                target[index].copy_(getattr(program, name)[0].float())
            coverage[index] = 1.0
    finally:
        inputs.video_store.close()
    if context.world_size > 1:
        for value in (*tensors.values(), coverage):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if not bool(torch.all(coverage == 1.0)):
        raise RuntimeError("free Program initialization did not cover every fit task")
    programs = {
        ordinal: ECPProgram(
            language=tensors["language"][index : index + 1],
            scene=tensors["scene"][index : index + 1],
            process=tensors["process"][index : index + 1],
            presence=tensors["presence"][index : index + 1],
            uncertainty=tensors["uncertainty"][index : index + 1],
        )
        for index, ordinal in enumerate(ordinals)
    }
    cell = config["free_program_oracle"]
    return TaskLocalFreeProgramTable(
        programs,
        process_delta_scale=float(cell["process_delta_scale"]),
        uncertainty_log_scale_bound=float(cell["uncertainty_log_scale_bound"]),
    ).to(context.device)
