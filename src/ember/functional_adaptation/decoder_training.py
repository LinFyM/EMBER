"""Shared task/code bookkeeping for fixed functional-decoder fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import load_task_expert_config
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.meta_contract import meta_expert_rows
from ember.functional_adaptation.decoder import (
    FunctionalAdapterDecoder,
    FunctionalCodebook,
)
from ember.functional_adaptation.decoder_flow_checkpoint import (
    RUN_SCHEMA as DECODER_RUN_SCHEMA,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_source_checkpoint import read_json


CONFIG_SCHEMA = "ember_pi05_functional_adapter_v1"


@dataclass(frozen=True)
class ExpertAdapterRecord:
    ordinal: int
    global_task_id: int
    language: str
    checkpoint: Path
    split_role: str = "train"


@dataclass(frozen=True)
class DecoderTaskSplit:
    fit: tuple[ExpertAdapterRecord, ...]
    held: tuple[ExpertAdapterRecord, ...]


@dataclass(frozen=True)
class MetaDecoderCodeTargets:
    decoder_checkpoint: Path
    train_codes: Mapping[int, torch.Tensor]
    held_codes: Mapping[int, torch.Tensor]


def load_functional_adapter_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    authorities = config.get("authorities", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "profile_candidate"
        or config.get("content_hash_policy") != "disabled_by_owner"
        or any(not (repo_root / str(value)).is_file() for value in authorities.values())
    ):
        raise ValueError("functional-adapter profile contract changed")
    return config


def authority_path(config: Mapping[str, Any], name: str, repo_root: Path) -> Path:
    return repo_root / str(config["authorities"][name])


def inspect_train24_expert_bank(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    source_run: Path,
    checkpoint: Path,
    bank_root: Path,
) -> dict[str, Any]:
    """Resolve the sealed bank once without creating a second bank reader."""

    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", repo_root), repo_root
    )
    source = inspect_source_checkpoint(
        authorities,
        source_run,
        checkpoint,
        evaluation_mode="formal",
    )
    manifest = read_json(authority_path(config, "target_data_manifest", repo_root))
    task_keys = [
        (str(row["suite"]), int(row["task_id"]))
        for row in manifest["tasks"]
        if row["split_role"] == "train"
    ]
    return inspect_task_expert_bank(
        config_path=authority_path(config, "train24_experts", repo_root),
        bank_root=bank_root,
        step=int(config["train24_mechanism"]["expert_step"]),
        source=source,
        task_keys=task_keys,
        evaluation_role="development_train",
        require_formal=True,
    )


def inspect_nonheld_meta_expert_bank(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    source_run: Path,
    checkpoint: Path,
    bank_root: Path,
) -> dict[str, Any]:
    """Resolve the complete 71-task bank under its fixed meta split."""

    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", repo_root), repo_root
    )
    source = inspect_source_checkpoint(
        authorities,
        source_run,
        checkpoint,
        evaluation_mode="formal",
    )
    expert_config_path = authority_path(config, "meta_experts", repo_root)
    expert_config = load_task_expert_config(expert_config_path)
    rows = meta_expert_rows(expert_config)
    return inspect_task_expert_bank(
        config_path=expert_config_path,
        bank_root=bank_root,
        step=int(config["production_meta"]["expert_step"]),
        source=source,
        task_keys=[(str(row["suite"]), int(row["task_id"])) for row in rows],
        evaluation_role="nonheld_meta",
        require_formal=True,
    )


def expert_records(bank: Mapping[str, Any]) -> tuple[ExpertAdapterRecord, ...]:
    records = tuple(
        sorted(
            (
                ExpertAdapterRecord(
                    ordinal=int(row["ordinal"]),
                    global_task_id=int(row["global_task_id"]),
                    language=str(row["language"]),
                    checkpoint=Path(str(row["checkpoint"])).resolve(),
                    split_role=str(row.get("split_role", "train")),
                )
                for row in bank["tasks"]
            ),
            key=lambda row: row.ordinal,
        )
    )
    if tuple(row.ordinal for row in records) != tuple(range(len(records))):
        raise ValueError("expert bank ordinals are not contiguous")
    return records


def decoder_task_split(
    records: Sequence[ExpertAdapterRecord], *, fold_count: int, held_out_fold: int
) -> DecoderTaskSplit:
    fit = tuple(row for row in records if row.ordinal % fold_count != held_out_fold)
    held = tuple(row for row in records if row.ordinal % fold_count == held_out_fold)
    if not fit or not held or len(fit) + len(held) != len(records):
        raise ValueError("functional-decoder task split is empty")
    return DecoderTaskSplit(fit=fit, held=held)


def meta_decoder_task_split(
    records: Sequence[ExpertAdapterRecord],
) -> DecoderTaskSplit:
    fit = tuple(row for row in records if row.split_role == "meta_train")
    held = tuple(
        row for row in records if row.split_role == "meta_validation_oracle"
    )
    if len(fit) != 56 or len(held) != 15 or len(fit) + len(held) != len(records):
        raise ValueError("non-held decoder split differs from the fixed 56/15 roles")
    return DecoderTaskSplit(fit=fit, held=held)


def load_expert_states(
    records: Sequence[ExpertAdapterRecord],
    contract: LoRAContract,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], ...]:
    result = []
    for record in records:
        state = load_file(str(record.checkpoint / "adapter.safetensors"), device="cpu")
        validate_lora_state(state, contract)
        result.append({name: value.to(device) for name, value in state.items()})
    return tuple(result)


def load_meta_decoder_code_targets(
    profile_root: Path, *, device: torch.device | str
) -> MetaDecoderCodeTargets:
    """Bind fixed, whitened codes to non-held task identities for supervision."""

    root = profile_root.resolve()
    result = read_json(root / "result.json")
    run_contract = result.get("run_contract", {})
    run = read_json(root / "run_contract.json")
    if (
        result.get("schema_version") != "ember_pi05_functional_flow_profile_v1"
        or result.get("surface") != "nonheld_meta"
        or result.get("mode") != "formal"
        or result.get("formal_authority") is not True
        or run_contract.get("schema_version") != DECODER_RUN_SCHEMA
        or run.get("schema_version") != DECODER_RUN_SCHEMA
        or run.get("mode") != "formal"
        or run.get("surface") != "nonheld_meta"
        or result.get("repository", {}).get("dirty_paths") != []
    ):
        raise ValueError("non-held decoder result is not a formal fixed-code authority")
    train_ids = tuple(int(value) for value in result["active_fit_global_task_ids"])
    held_ids = tuple(int(value) for value in result["active_held_global_task_ids"])
    decoder_path = root / "decoder.safetensors"
    state = load_file(str(decoder_path), device=str(device))
    train = state.get("codebook.weight")
    held = load_file(str(root / "held_codes.safetensors"), device=str(device)).get(
        "held_codes"
    )
    if (
        train is None
        or held is None
        or train.ndim != 2
        or held.ndim != 2
        or len(train_ids) != 56
        or train.shape[0] != len(train_ids)
        or len(held_ids) != 15
        or held.shape[0] != len(held_ids)
        or train.shape[1] != held.shape[1]
    ):
        raise ValueError("non-held fixed-code checkpoint changed shape")
    return MetaDecoderCodeTargets(
        decoder_checkpoint=decoder_path,
        train_codes={task_id: train[index] for index, task_id in enumerate(train_ids)},
        held_codes={task_id: held[index] for index, task_id in enumerate(held_ids)},
    )


class FunctionalDecoderSystem(torch.nn.Module):
    """One privileged codebook and one shared complete-LoRA decoder."""

    def __init__(
        self,
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        *,
        task_count: int,
        code_width: int,
        address_width: int,
        hidden_width: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.codebook = FunctionalCodebook(task_count, code_width, seed=seed)
        self.decoder = FunctionalAdapterDecoder(
            contract,
            template_state,
            code_width=code_width,
            address_width=address_width,
            hidden_width=hidden_width,
            initialization_seed=seed,
        )

    def forward(self, task_index: int) -> dict[str, torch.Tensor]:
        index = torch.tensor(task_index, device=self.codebook.weight.device)
        return self.decoder(self.codebook(index))


def balanced_task_order(task_count: int, steps: int, *, seed: int) -> tuple[int, ...]:
    """Produce task-equal shuffled macros without outcome-based sampling."""

    if task_count <= 0 or steps <= 0:
        raise ValueError("balanced task order requires positive sizes")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order: list[int] = []
    while len(order) < steps:
        order.extend(torch.randperm(task_count, generator=generator).tolist())
    return tuple(order[:steps])
