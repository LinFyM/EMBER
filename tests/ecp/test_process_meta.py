from pathlib import Path

import numpy as np
import pytest

from ember.ecp.process_meta import (
    ProcessMetaError,
    ProcessMetaFamily,
    ProcessVariant,
    TemporalPredicateOrderEnv,
    _variant_phase_expert_authorities,
)


class _PredicateEnv:
    def __init__(self) -> None:
        self.values = {"red": False, "yellow": False}

    def _eval_predicate(self, predicate: list[str]) -> bool:
        return self.values[predicate[1]]

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        transition = int(action[0])
        if transition == 1:
            self.values["red"] = True
        elif transition == 2:
            self.values["red"] = False
        elif transition == 3:
            self.values["yellow"] = True
        elif transition == 4:
            self.values["red"] = True
        return {}, 0.0, False, {}


def _family(tmp_path: Path) -> ProcessMetaFamily:
    return ProcessMetaFamily(
        family_id="separate_plates",
        exact_language="put both mugs on their plates",
        bddl_path=tmp_path / "task.bddl",
        init_states_path=tmp_path / "init.pt",
        base_task_suite="libero_90",
        base_task_id=65,
        init_state_ids=tuple(range(50)),
        predicates={"red": ("on", "red", "left"), "yellow": ("on", "yellow", "right")},
        phase_languages={"red": "put red", "yellow": "put yellow"},
        variants=(
            ProcessVariant("red_then_yellow", ("red", "yellow")),
            ProcessVariant("yellow_then_red", ("yellow", "red")),
        ),
    )


def _record(
    tmp_path: Path, name: str, *, task_id: int, language: str, role: str
) -> dict[str, object]:
    checkpoint = tmp_path / name
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(name.encode())
    return {
        "task_id": task_id,
        "language": language,
        "checkpoint": name,
        "adapter_bytes": len(name),
        "step": 1000,
        "role": role,
    }


def test_variant_phase_recovery_authority_fixes_primitive_then_recovery(
    tmp_path: Path,
) -> None:
    teacher = {
        "variant_phase_experts": {
            "red_then_yellow": {
                "red": _record(
                    tmp_path,
                    "primitive_red",
                    task_id=65,
                    language="put red",
                    role="original_primitive_first",
                ),
                "yellow": _record(
                    tmp_path,
                    "recovery_yellow",
                    task_id=68,
                    language="put yellow",
                    role="composite_context_recovery_second",
                ),
            },
            "yellow_then_red": {
                "yellow": _record(
                    tmp_path,
                    "primitive_yellow",
                    task_id=68,
                    language="put yellow",
                    role="original_primitive_first",
                ),
                "red": _record(
                    tmp_path,
                    "recovery_red",
                    task_id=65,
                    language="put red",
                    role="composite_context_recovery_second",
                ),
            },
        }
    }
    routes = _variant_phase_expert_authorities(
        teacher, repo_root=tmp_path, family=_family(tmp_path)
    )
    assert routes["red_then_yellow"]["red"].role == "original_primitive_first"
    assert (
        routes["red_then_yellow"]["yellow"].role == "composite_context_recovery_second"
    )
    assert routes["yellow_then_red"]["yellow"].task_id == 68


def test_variant_phase_recovery_rejects_reversed_roles(tmp_path: Path) -> None:
    teacher = {
        "variant_phase_experts": {
            "red_then_yellow": {
                "red": _record(
                    tmp_path,
                    "bad_red",
                    task_id=65,
                    language="put red",
                    role="composite_context_recovery_second",
                ),
                "yellow": _record(
                    tmp_path,
                    "bad_yellow",
                    task_id=68,
                    language="put yellow",
                    role="original_primitive_first",
                ),
            },
            "yellow_then_red": {
                "yellow": _record(
                    tmp_path,
                    "primitive_yellow",
                    task_id=68,
                    language="put yellow",
                    role="original_primitive_first",
                ),
                "red": _record(
                    tmp_path,
                    "recovery_red",
                    task_id=65,
                    language="put red",
                    role="composite_context_recovery_second",
                ),
            },
        }
    }
    with pytest.raises(ProcessMetaError, match="authority changed"):
        _variant_phase_expert_authorities(
            teacher, repo_root=tmp_path, family=_family(tmp_path)
        )


def test_temporal_wrapper_records_post_completion_predicate_drop() -> None:
    env = TemporalPredicateOrderEnv(
        _PredicateEnv(),
        predicates={"red": ("on", "red", "left"), "yellow": ("on", "yellow", "right")},
        required_order=("red", "yellow"),
    )
    env.begin_episode()
    for transition in (1, 2, 3, 4):
        env.step(np.asarray([transition]))
    snapshot = env.snapshot()
    assert snapshot["success"] is True
    assert snapshot["post_completion_drop_steps"]["red"] == (2,)
