"""Strict language-only role factors for the pinned LIBERO-90 specification surface."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


FACTOR_SCHEMA = "libero90_role_factors_v1"
SCENE_PATTERN = re.compile(r"^[A-Z]+(?:_[A-Z]+)*_SCENE\d+$")


class FactorParseError(ValueError):
    """Raised when a task specification is outside or ambiguous under the sealed grammar."""


# Locatives attached to these noun phrases are source selectors. They are not
# destination relations. Values are (canonical mobile object, source selector).
OBJECT_PHRASES: dict[str, tuple[str, str | None]] = {
    "the alphabet soup": ("alphabet_soup", None),
    "the back black bowl": ("black_bowl", "back"),
    "the black bowl": ("black_bowl", None),
    "the black bowl at the back": ("black_bowl", "back"),
    "the black bowl at the front": ("black_bowl", "front"),
    "the black bowl in the middle": ("black_bowl", "middle"),
    "the black bowl on the left": ("black_bowl", "left"),
    "the book": ("book", None),
    "the book in the middle": ("book", "middle"),
    "the book on the left": ("book", "left"),
    "the book on the right": ("book", "right"),
    "the bowl": ("bowl", None),
    "the butter": ("butter", None),
    "the butter at the back": ("butter", "back"),
    "the butter at the front": ("butter", "front"),
    "the chocolate pudding": ("chocolate_pudding", None),
    "the cream cheese": ("cream_cheese", None),
    "the cream cheese box": ("cream_cheese_box", None),
    "the frying pan": ("frying_pan", None),
    "the ketchup": ("ketchup", None),
    "the left bowl": ("bowl", "left"),
    "the middle black bowl": ("black_bowl", "middle"),
    "the milk": ("milk", None),
    "the moka pot": ("moka_pot", None),
    "the orange juice": ("orange_juice", None),
    "the red mug": ("red_mug", None),
    "the right bowl": ("bowl", "right"),
    "the right moka pot": ("moka_pot", "right"),
    "the salad dressing": ("salad_dressing", None),
    "the tomato sauce": ("tomato_sauce", None),
    "the white bowl": ("white_bowl", None),
    "the white mug": ("white_mug", None),
    "the wine bottle": ("wine_bottle", None),
    "the yellow and white mug": ("yellow_white_mug", None),
}


# Values are (target relation, canonical target receptacle, target selectors).
DESTINATIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "in the top drawer of the cabinet": ("in", "cabinet_drawer", ("top",)),
    "in the bottom drawer of the cabinet": ("in", "cabinet_drawer", ("bottom",)),
    "on the plate": ("on", "plate", ()),
    "on top of the cabinet": ("on_top_of", "cabinet", ()),
    "on the stove": ("on", "stove", ()),
    "on the wine rack": ("on", "wine_rack", ()),
    "to the front of the white mug": ("front_of", "white_mug", ()),
    "to the right of the plate": ("right_of", "plate", ()),
    "to the left of the plate": ("left_of", "plate", ()),
    "on the cabinet shelf": ("on", "cabinet_shelf", ()),
    "under the cabinet shelf": ("under", "cabinet_shelf", ()),
    "in the basket": ("in", "basket", ()),
    "in the tray": ("in", "tray", ()),
    "on the left plate": ("on", "plate", ("left",)),
    "on the right plate": ("on", "plate", ("right",)),
    "in the front compartment of the caddy": ("in", "caddy", ("front_compartment",)),
    "in the left compartment of the caddy": ("in", "caddy", ("left_compartment",)),
    "in the right compartment of the caddy": ("in", "caddy", ("right_compartment",)),
    "in the back compartment of the caddy": ("in", "caddy", ("back_compartment",)),
    "to the right of the caddy": ("right_of", "caddy", ()),
    "on top of the shelf": ("on_top_of", "cabinet_shelf", ()),
}


FACTOR_ROLE_DEFINITIONS = {
    "verb": "normalized explicit operation; put/place collapse to place",
    "moved_object": "mobile grammatical patient, never a scene distractor",
    "target_receptacle": "grammatical destination entity or supporting object",
    "target_relation": "normalized destination relation",
    "source_selector": "locative that identifies which movable instance is acted on",
    "target_selector": "task-relevant destination region or target-instance selector",
    "actuated_fixture": "fixture directly opened, closed, or toggled",
    "actuated_selector": "task-relevant subregion of an actuated fixture",
    "order_composition": (
        "steps follow explicit conjunction order; pick-up/place and stack/group-place are two steps"
    ),
}


def _step(
    order: int,
    verb: str,
    *,
    moved_objects: Iterable[str] = (),
    source_selectors: Iterable[str] = (),
    target_receptacle: str | None = None,
    target_relation: str | None = None,
    target_selectors: Iterable[str] = (),
    actuated_fixture: str | None = None,
    actuated_selectors: Iterable[str] = (),
    group_size: int = 1,
) -> dict[str, Any]:
    return {
        "order": order,
        "verb": verb,
        "moved_objects": list(moved_objects),
        "source_selectors": list(source_selectors),
        "target_receptacle": target_receptacle,
        "target_relation": target_relation,
        "target_selectors": list(target_selectors),
        "actuated_fixture": actuated_fixture,
        "actuated_selectors": list(actuated_selectors),
        "group_size": group_size,
    }


def _parse_object_exact(phrase: str) -> tuple[str, tuple[str, ...]]:
    result = OBJECT_PHRASES.get(phrase)
    if result is None:
        raise FactorParseError(f"unknown moved-object phrase: {phrase!r}")
    moved_object, selector = result
    return moved_object, (() if selector is None else (selector,))


def _parse_destination_exact(phrase: str) -> tuple[str, str, tuple[str, ...]]:
    result = DESTINATIONS.get(phrase)
    if result is None:
        raise FactorParseError(f"unknown destination phrase: {phrase!r}")
    return result


def _parse_place_body(body: str, *, order: int = 0, group_size: int = 1) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for phrase, (moved_object, selector) in OBJECT_PHRASES.items():
        prefix = f"{phrase} "
        if not body.startswith(prefix):
            continue
        destination = DESTINATIONS.get(body[len(prefix) :])
        if destination is None:
            continue
        relation, receptacle, target_selectors = destination
        candidates.append(
            _step(
                order,
                "place",
                moved_objects=(moved_object,),
                source_selectors=(() if selector is None else (selector,)),
                target_receptacle=receptacle,
                target_relation=relation,
                target_selectors=target_selectors,
                group_size=group_size,
            )
        )
    if len(candidates) != 1:
        label = "unknown" if not candidates else "ambiguous"
        raise FactorParseError(f"{label} place clause: {body!r}")
    return candidates[0]


def _parse_atomic_actuation(language: str) -> dict[str, Any] | None:
    drawer = re.fullmatch(r"(open|close) the (top|bottom) drawer of the cabinet", language)
    if drawer is not None:
        verb, selector = drawer.groups()
        return {"steps": [_step(0, verb, actuated_fixture="cabinet_drawer", actuated_selectors=(selector,))]}
    appliance = re.fullmatch(r"(open|close) the microwave", language)
    if appliance is not None:
        return {"steps": [_step(0, appliance.group(1), actuated_fixture="microwave")]}
    stove = re.fullmatch(r"turn (on|off) the stove", language)
    if stove is not None:
        return {"steps": [_step(0, f"turn_{stove.group(1)}", actuated_fixture="stove")]}
    return None


def _parse_compound(language: str) -> dict[str, Any] | None:
    if language == "close the top drawer of the cabinet and put the black bowl on top of it":
        return {
            "steps": [
                _step(0, "close", actuated_fixture="cabinet_drawer", actuated_selectors=("top",)),
                _parse_place_body("the black bowl on top of the cabinet", order=1),
            ]
        }
    if language == "open the top drawer of the cabinet and put the bowl in it":
        return {
            "steps": [
                _step(0, "open", actuated_fixture="cabinet_drawer", actuated_selectors=("top",)),
                _parse_place_body("the bowl in the top drawer of the cabinet", order=1),
            ]
        }
    if language == "turn on the stove and put the frying pan on it":
        return {
            "steps": [
                _step(0, "turn_on", actuated_fixture="stove"),
                _parse_place_body("the frying pan on the stove", order=1),
            ]
        }
    if language == "close the bottom drawer of the cabinet and open the top drawer":
        return {
            "steps": [
                _step(0, "close", actuated_fixture="cabinet_drawer", actuated_selectors=("bottom",)),
                _step(1, "open", actuated_fixture="cabinet_drawer", actuated_selectors=("top",)),
            ]
        }
    suffix = " and close it"
    if language.startswith("put ") and language.endswith(suffix):
        placement = _parse_place_body(language[len("put ") : -len(suffix)], order=0)
        if placement["target_receptacle"] != "cabinet_drawer" or placement[
            "target_selectors"
        ] != ["top"]:
            raise FactorParseError("unknown close-it antecedent outside the top drawer grammar")
        return {
            "steps": [
                placement,
                _step(1, "close", actuated_fixture="cabinet_drawer", actuated_selectors=("top",)),
            ]
        }
    return None


def _parse_pick_place(language: str) -> dict[str, Any] | None:
    matched = re.fullmatch(r"pick up (.+) and (?:put|place) it (.+)", language)
    if matched is None:
        return None
    moved_object, source_selectors = _parse_object_exact(matched.group(1))
    relation, receptacle, target_selectors = _parse_destination_exact(matched.group(2))
    return {
        "steps": [
            _step(
                0,
                "pick_up",
                moved_objects=(moved_object,),
                source_selectors=source_selectors,
            ),
            _step(
                1,
                "place",
                moved_objects=(moved_object,),
                source_selectors=source_selectors,
                target_receptacle=receptacle,
                target_relation=relation,
                target_selectors=target_selectors,
            ),
        ]
    }


def _parse_stack(language: str) -> dict[str, Any] | None:
    matched = re.fullmatch(r"stack (.+?) on (.+?)(?: and place them (.+))?", language)
    if matched is None:
        return None
    moved_object, source_selectors = _parse_object_exact(matched.group(1))
    target_object, target_selectors = _parse_object_exact(matched.group(2))
    steps = [
        _step(
            0,
            "stack",
            moved_objects=(moved_object,),
            source_selectors=source_selectors,
            target_receptacle=target_object,
            target_relation="on",
            target_selectors=target_selectors,
        )
    ]
    group_destination = matched.group(3)
    if group_destination is not None:
        relation, receptacle, selectors = _parse_destination_exact(group_destination)
        steps.append(
            _step(
                1,
                "place",
                moved_objects=(moved_object,),
                target_receptacle=receptacle,
                target_relation=relation,
                target_selectors=selectors,
                group_size=2,
            )
        )
    return {"steps": steps}


def _parse_single_place(language: str) -> dict[str, Any] | None:
    if not language.startswith("put "):
        return None
    try:
        return {"steps": [_parse_place_body(language[len("put ") :])]}
    except FactorParseError as error:
        if " and " in language:
            return None
        raise error


def require_single_parse(language: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise FactorParseError(f"unknown LIBERO-90 instruction template: {language!r}")
    if len(candidates) != 1:
        raise FactorParseError(f"ambiguous LIBERO-90 instruction template: {language!r}")
    return candidates[0]


def _primitive_role_atoms(steps: list[dict[str, Any]]) -> list[str]:
    atoms: set[str] = set()
    for step in steps:
        atoms.add(f"verb:{step['verb']}")
        atoms.update(f"moved_object:{value}" for value in step["moved_objects"])
        atoms.update(f"source_selector:{value}" for value in step["source_selectors"])
        if step["target_receptacle"] is not None:
            atoms.add(f"target_receptacle:{step['target_receptacle']}")
        if step["target_relation"] is not None:
            atoms.add(f"target_relation:{step['target_relation']}")
        atoms.update(f"target_selector:{value}" for value in step["target_selectors"])
        if step["actuated_fixture"] is not None:
            atoms.add(f"actuated_fixture:{step['actuated_fixture']}")
        atoms.update(f"actuated_selector:{value}" for value in step["actuated_selectors"])
    return sorted(atoms)


def parse_task_language(language: str) -> dict[str, Any]:
    if not isinstance(language, str) or language != " ".join(language.split()) or language.lower() != language:
        raise FactorParseError("unknown non-canonical language normalization")
    candidates = []
    for parser in (
        _parse_atomic_actuation,
        _parse_compound,
        _parse_pick_place,
        _parse_stack,
        _parse_single_place,
    ):
        parsed = parser(language)
        if parsed is not None:
            candidates.append(parsed)
    factors = require_single_parse(language, candidates)
    steps = factors["steps"]
    signature_payload = [
        {key: value for key, value in step.items() if key != "order"} for step in steps
    ]
    return {
        "factor_schema": FACTOR_SCHEMA,
        "steps": steps,
        "primitive_role_atoms": _primitive_role_atoms(steps),
        "order_signature": ">".join(step["verb"] for step in steps),
        "composition_signature": json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ),
    }


def factor_task(*, task_index: int, scene: str, language: str) -> dict[str, Any]:
    if not isinstance(task_index, int) or task_index < 0:
        raise FactorParseError("task index must be a non-negative integer")
    if not isinstance(scene, str) or SCENE_PATTERN.fullmatch(scene) is None:
        raise FactorParseError(f"unknown scene identity: {scene!r}")
    factors = parse_task_language(language)
    return {
        "task_index": task_index,
        "scene": scene,
        "language": language,
        **factors,
        "difficulty": {
            "operation_count": len(factors["steps"]),
            "composition_depth": len(factors["steps"]),
        },
    }
