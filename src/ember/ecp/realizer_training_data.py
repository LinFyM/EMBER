"""Task-equal in-memory batches for the fixed ECP effect realizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.ecp.realizer_code import EFFECT_CODE_AUTHORITY_SCHEMA
from ember.ecp.realizer_evidence import load_effect_member
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_source_checkpoint import read_json


@dataclass(frozen=True)
class EffectCodeBatch:
    code: torch.Tensor
    particle_mask: torch.Tensor
    reliability: torch.Tensor
    targets: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    member_indices: tuple[int, ...]


class PackedEffectCodeDataset:
    def __init__(
        self,
        *,
        manifest_path: Path,
        contract: LoRAContract,
        device: torch.device,
        include_held: bool = True,
    ) -> None:
        manifest = read_json(manifest_path.resolve())
        rows = tuple(dict(row) for row in manifest.get("members", ()))
        if (
            manifest.get("schema_version") != EFFECT_CODE_AUTHORITY_SCHEMA
            or manifest.get("status") != "complete_fit_only_effect_code_coordinate"
            or len(rows) != 118
        ):
            raise ValueError("fixed effect code training authority changed")
        fit_indices = tuple(
            index for index, row in enumerate(rows) if row["fold_role"] == "fit"
        )
        held_indices = tuple(
            index
            for index, row in enumerate(rows)
            if row["fold_role"] == "held_transform_only"
        )
        groups: dict[str, list[int]] = {}
        for index in fit_indices:
            groups.setdefault(str(rows[index]["asset_key"]), []).append(index)
        if (
            len(groups) != int(manifest["fit_tasks"])
            or len(held_indices) != int(manifest["held_members"])
            or len({rows[index]["asset_key"] for index in held_indices})
            != int(manifest["held_tasks"])
        ):
            raise ValueError("fixed effect code fold roles changed")

        loaded_indices = tuple(range(len(rows))) if include_held else fit_indices
        original_to_local = {
            original: local for local, original in enumerate(loaded_indices)
        }
        codes, masks, reliabilities = [], [], []
        target_a: list[list[torch.Tensor]] = [[] for _ in contract.targets]
        target_b: list[list[torch.Tensor]] = [[] for _ in contract.targets]
        for original_index in loaded_indices:
            row = rows[original_index]
            code = load_file(str(Path(str(row["effect_code_path"])).resolve()))[
                "effect_code"
            ]
            if code.ndim != 4 or code.shape[1:] != (8, 38, 128) or code.shape[0] > 4:
                raise ValueError("fixed effect code member changed shape")
            padded = torch.zeros(4, 8, 38, 128, dtype=code.dtype)
            padded[: code.shape[0]] = code
            mask = torch.arange(4) < code.shape[0]
            _, residual, _, _ = load_effect_member(
                Path(str(row["tensor_path"])), contract=contract
            )
            codes.append(padded)
            masks.append(mask)
            reliabilities.append(float(row["reliability"]))
            for target_index, target in enumerate(contract.targets):
                target_a[target_index].append(
                    residual[target.name + LORA_A_SUFFIX]
                )
                target_b[target_index].append(
                    residual[target.name + LORA_B_SUFFIX]
                )
        self.rows = tuple(rows[index] for index in loaded_indices)
        self.fit_indices = tuple(original_to_local[index] for index in fit_indices)
        self.held_indices = tuple(
            original_to_local[index] for index in held_indices if index in original_to_local
        )
        self.task_groups = tuple(
            tuple(original_to_local[index] for index in groups[key])
            for key in sorted(groups)
        )
        self.code = torch.stack(codes).to(device)
        self.particle_mask = torch.stack(masks).to(device)
        self.reliability = torch.tensor(reliabilities, device=device)
        self.target_a = tuple(torch.stack(values).to(device) for values in target_a)
        self.target_b = tuple(torch.stack(values).to(device) for values in target_b)

    def training_batch(self, step: int) -> EffectCodeBatch:
        indices = tuple(
            members[step % len(members)] for members in self.task_groups
        )
        index = torch.tensor(indices, device=self.code.device)
        return self._batch(index, indices)

    def held_batch(self, member: str = "latest") -> EffectCodeBatch:
        by_task: dict[str, list[int]] = {}
        for index in self.held_indices:
            by_task.setdefault(str(self.rows[index]["asset_key"]), []).append(index)
        selected = []
        for key in sorted(by_task):
            matches = [
                index
                for index in by_task[key]
                if str(self.rows[index]["member"]) == member
            ]
            if len(matches) != 1:
                raise ValueError("held effect code lacks the fixed member lineage")
            selected.append(matches[0])
        indices = tuple(selected)
        index = torch.tensor(indices, device=self.code.device)
        return self._batch(index, indices)

    def _batch(self, index: torch.Tensor, indices: tuple[int, ...]) -> EffectCodeBatch:
        return EffectCodeBatch(
            code=self.code.index_select(0, index),
            particle_mask=self.particle_mask.index_select(0, index),
            reliability=self.reliability.index_select(0, index),
            targets=tuple(
                (
                    self.target_a[target].index_select(0, index),
                    self.target_b[target].index_select(0, index),
                )
                for target in range(len(self.target_a))
            ),
            member_indices=indices,
        )
