from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.libero_task_factors import (  # noqa: E402
    FactorParseError,
    factor_task,
    parse_task_language,
    require_single_parse,
)


class LiberoTaskFactorParserTest(unittest.TestCase):
    def test_drawer_sequence_preserves_roles_and_order(self) -> None:
        factors = parse_task_language(
            "put the butter at the back in the top drawer of the cabinet and close it"
        )

        self.assertEqual([step["verb"] for step in factors["steps"]], ["place", "close"])
        self.assertEqual(factors["steps"][0]["moved_objects"], ["butter"])
        self.assertEqual(factors["steps"][0]["source_selectors"], ["back"])
        self.assertEqual(factors["steps"][0]["target_receptacle"], "cabinet_drawer")
        self.assertEqual(factors["steps"][0]["target_relation"], "in")
        self.assertEqual(factors["steps"][0]["target_selectors"], ["top"])
        self.assertEqual(factors["steps"][1]["actuated_fixture"], "cabinet_drawer")
        self.assertEqual(factors["steps"][1]["actuated_selectors"], ["top"])

    def test_stack_then_group_place_is_two_explicit_steps(self) -> None:
        factors = parse_task_language(
            "stack the left bowl on the right bowl and place them in the tray"
        )

        self.assertEqual([step["verb"] for step in factors["steps"]], ["stack", "place"])
        self.assertEqual(factors["steps"][0]["moved_objects"], ["bowl"])
        self.assertEqual(factors["steps"][0]["source_selectors"], ["left"])
        self.assertEqual(factors["steps"][0]["target_receptacle"], "bowl")
        self.assertEqual(factors["steps"][0]["target_selectors"], ["right"])
        self.assertEqual(factors["steps"][1]["moved_objects"], ["bowl"])
        self.assertEqual(factors["steps"][1]["group_size"], 2)
        self.assertEqual(factors["steps"][1]["target_receptacle"], "tray")

    def test_target_relation_is_not_confused_with_source_selector(self) -> None:
        source_selected = parse_task_language("put the black bowl at the front on the plate")
        relative_target = parse_task_language(
            "put the yellow and white mug to the front of the white mug"
        )

        self.assertIn("source_selector:front", source_selected["primitive_role_atoms"])
        self.assertNotIn("target_relation:front_of", source_selected["primitive_role_atoms"])
        self.assertIn("target_relation:front_of", relative_target["primitive_role_atoms"])
        self.assertNotIn("source_selector:front", relative_target["primitive_role_atoms"])

    def test_pick_place_compartment_roles_are_explicit(self) -> None:
        factors = parse_task_language(
            "pick up the book and place it in the front compartment of the caddy"
        )

        self.assertEqual([step["verb"] for step in factors["steps"]], ["pick_up", "place"])
        self.assertEqual(factors["steps"][1]["target_receptacle"], "caddy")
        self.assertEqual(factors["steps"][1]["target_relation"], "in")
        self.assertEqual(factors["steps"][1]["target_selectors"], ["front_compartment"])

    def test_unknown_and_ambiguous_parses_fail_closed(self) -> None:
        with self.assertRaisesRegex(FactorParseError, "unknown"):
            parse_task_language("wave at the camera")
        with self.assertRaisesRegex(FactorParseError, "ambiguous"):
            require_single_parse("synthetic ambiguous instruction", [{"steps": []}, {"steps": []}])

    def test_factor_task_binds_only_allowed_identity_fields(self) -> None:
        record = factor_task(
            task_index=34,
            scene="KITCHEN_SCENE6",
            language="put the yellow and white mug to the front of the white mug",
        )

        self.assertEqual(record["task_index"], 34)
        self.assertEqual(record["scene"], "KITCHEN_SCENE6")
        self.assertEqual(record["difficulty"], {"operation_count": 1, "composition_depth": 1})
        self.assertNotIn("bddl", record)
        self.assertNotIn("reward", record)
        self.assertNotIn("action", record)


if __name__ == "__main__":
    unittest.main()
