"""Process-identifying meta-task contracts and temporal reward semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROCESS_META_MANIFEST_SCHEMA = "ember_ecp_process_meta_manifest_v1"


class ProcessMetaError(RuntimeError):
    """Raised when process-meta data would violate its information contract."""


@dataclass(frozen=True)
class ProcessVariant:
    name: str
    required_order: tuple[str, str]


@dataclass(frozen=True)
class ProcessMetaFamily:
    family_id: str
    exact_language: str
    bddl_path: Path
    init_states_path: Path
    init_state_ids: tuple[int, ...]
    predicates: Mapping[str, tuple[str, ...]]
    phase_languages: Mapping[str, str]
    variants: tuple[ProcessVariant, ProcessVariant]

    def variant(self, name: str) -> ProcessVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise ProcessMetaError(f"unknown process-meta variant: {name}")


@dataclass(frozen=True)
class ProcessMetaAuthority:
    source_evaluation_root: Path
    normalization_path: Path
    tokenizer_path: Path
    family: ProcessMetaFamily
    rollout: Mapping[str, Any]
    information_wall: Mapping[str, Any]


def load_process_meta_authority(
    manifest_path: Path,
    *,
    repo_root: Path,
    libero_init_root: Path,
) -> ProcessMetaAuthority:
    """Resolve the one active minimal-pair family without benchmark routing."""

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != PROCESS_META_MANIFEST_SCHEMA
        or value.get("status") != "active_minimal_pair_feasibility"
    ):
        raise ProcessMetaError("process-meta manifest is not the active feasibility contract")
    raw = value["family"]
    bddl_path = repo_root / str(raw["bddl"]["path"])
    init = raw["base_init_states"]
    init_path = (
        libero_init_root / str(init["suite"]) / str(init["filename"])
    )
    if (
        not bddl_path.is_file()
        or bddl_path.stat().st_size != int(raw["bddl"]["bytes"])
        or not init_path.is_file()
        or init_path.stat().st_size != int(init["bytes"])
    ):
        raise ProcessMetaError("process-meta BDDL or base init-state authority changed")
    predicates = {
        str(name): tuple(str(item) for item in predicate)
        for name, predicate in raw["predicates"].items()
    }
    phase_languages = {
        str(name): str(language).strip()
        for name, language in raw["privileged_teacher_phase_languages"].items()
    }
    variants = tuple(
        ProcessVariant(
            name=str(row["name"]),
            required_order=tuple(str(item) for item in row["required_order"]),
        )
        for row in raw["variants"]
    )
    keys = tuple(predicates)
    if (
        len(keys) != 2
        or set(phase_languages) != set(keys)
        or len(variants) != 2
        or any(set(variant.required_order) != set(keys) for variant in variants)
        or variants[0].required_order != tuple(reversed(variants[1].required_order))
        or not str(raw["exact_language"]).strip()
    ):
        raise ProcessMetaError("process-meta minimal pair is not symmetric")
    ids = tuple(int(item) for item in init["state_ids"])
    if ids != tuple(range(int(init["count"]))):
        raise ProcessMetaError("process-meta init-state panel changed")
    family = ProcessMetaFamily(
        family_id=str(raw["family_id"]),
        exact_language=str(raw["exact_language"]).strip(),
        bddl_path=bddl_path,
        init_states_path=init_path,
        init_state_ids=ids,
        predicates=predicates,
        phase_languages=phase_languages,
        variants=(variants[0], variants[1]),
    )
    result = ProcessMetaAuthority(
        source_evaluation_root=repo_root / str(value["source_policy_authority"]),
        normalization_path=repo_root / str(value["normalization_authority"]),
        tokenizer_path=repo_root / str(value["tokenizer_authority"]),
        family=family,
        rollout=dict(value["rollout"]),
        information_wall=dict(value["information_wall"]),
    )
    if not all(
        path.is_file() if path.suffix else path.is_dir()
        for path in (
            result.source_evaluation_root,
            result.normalization_path,
            result.tokenizer_path,
        )
    ):
        raise ProcessMetaError("process-meta source policy authority is incomplete")
    return result


class TemporalPredicateOrderEnv:
    """Replace final-predicate success with an irreversible event-order contract."""

    def __init__(
        self,
        env: Any,
        *,
        predicates: Mapping[str, Sequence[str]],
        required_order: Sequence[str],
    ) -> None:
        self.env = env
        self.predicates = {
            str(name): tuple(str(item) for item in predicate)
            for name, predicate in predicates.items()
        }
        self.required_order = tuple(str(item) for item in required_order)
        if len(self.required_order) != 2 or set(self.required_order) != set(self.predicates):
            raise ProcessMetaError("temporal wrapper requires one two-event permutation")
        self.steps = 0
        self.invalid = False
        self.invalid_reason: str | None = None
        self._completed: dict[str, int] = {}
        self._last_values = {name: False for name in self.predicates}
        self._begun = False

    @property
    def _predicate_owner(self) -> Any:
        owner = getattr(self.env, "env", self.env)
        if not callable(getattr(owner, "_eval_predicate", None)):
            raise ProcessMetaError("LIBERO environment exposes no predicate evaluator")
        return owner

    def _values(self) -> dict[str, bool]:
        owner = self._predicate_owner
        return {
            name: bool(owner._eval_predicate(list(predicate)))
            for name, predicate in self.predicates.items()
        }

    def begin_episode(self) -> None:
        self.steps = 0
        self._completed = {}
        self._last_values = self._values()
        self._begun = True
        initially_true = [name for name, value in self._last_values.items() if value]
        self.invalid = bool(initially_true)
        self.invalid_reason = (
            "predicate_true_at_episode_start" if initially_true else None
        )

    @property
    def phase_key(self) -> str | None:
        if self.invalid:
            return None
        if len(self._completed) < len(self.required_order):
            return self.required_order[len(self._completed)]
        values = self._values()
        return next((name for name in self.required_order if not values[name]), None)

    @property
    def success(self) -> bool:
        if not self._begun:
            return False
        values = self._values()
        return (
            not self.invalid
            and len(self._completed) == len(self.required_order)
            and all(values.values())
        )

    def snapshot(self) -> dict[str, Any]:
        values = self._values() if self._begun else dict(self._last_values)
        return {
            "steps": self.steps,
            "predicate_values": values,
            "completion_steps": dict(self._completed),
            "phase_key": self.phase_key,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason,
            "success": self.success,
        }

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
        if not self._begun:
            raise ProcessMetaError("temporal wrapper stepped before reset")
        observation, _, _, info = self.env.step(action)
        self.steps += 1
        values = self._values()
        rising = [
            name
            for name, value in values.items()
            if value and not self._last_values[name] and name not in self._completed
        ]
        if not self.invalid and rising:
            expected = self.required_order[len(self._completed)]
            if len(rising) != 1 or rising[0] != expected:
                self.invalid = True
                self.invalid_reason = "wrong_or_simultaneous_first_satisfaction"
            else:
                self._completed[rising[0]] = self.steps
        self._last_values = values
        success = (
            not self.invalid
            and len(self._completed) == len(self.required_order)
            and all(values.values())
        )
        enriched = dict(info)
        enriched["process_meta"] = self.snapshot()
        return observation, float(success), success, enriched

    def reset(self) -> Any:
        observation = self.env.reset()
        self.begin_episode()
        return observation

    def set_init_state(self, state: np.ndarray) -> Any:
        observation = self.env.set_init_state(state)
        self.begin_episode()
        return observation

    def seed(self, seed: int) -> Any:
        return self.env.seed(seed)

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
