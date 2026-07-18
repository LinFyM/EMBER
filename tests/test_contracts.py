from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.contracts import ContractError, load_contract, validate_contract  # noqa: E402


class Phase0ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(ROOT / "configs" / "phase0.toml")

    def test_checked_in_contract_is_valid(self) -> None:
        validate_contract(self.contract)

    def test_gpu_ceiling_is_hard(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["resources"]["max_concurrent_gpus"] = 5
        with self.assertRaisesRegex(ContractError, "four GPU"):
            validate_contract(changed)

    def test_task_split_is_exact_and_disjoint(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["splits"]["validation"][0] = changed["splits"]["held_out"][0]
        with self.assertRaisesRegex(ContractError, "task split"):
            validate_contract(changed)

    def test_split_reseal_is_permanent_and_preserves_prior_split(self) -> None:
        reseal = self.contract["split_reseal"]
        self.assertEqual(reseal["minimum_source_role_occurrences"], 2)
        self.assertEqual(reseal["search_seed"], 20_260_718)
        self.assertNotEqual(reseal["prior_split"], self.contract["splits"])

        changed = copy.deepcopy(self.contract)
        changed["split_reseal"]["search_seed"] += 1
        with self.assertRaisesRegex(ContractError, "seed"):
            validate_contract(changed)

    def test_episode_authority_is_disjoint_and_complete(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["episode_authority"]["oracle_support"] = [27, 39]
        with self.assertRaisesRegex(ContractError, "episode authority"):
            validate_contract(changed)

    def test_external_revisions_are_immutable_hashes(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["upstreams"]["lerobot"]["commit"] = "main"
        with self.assertRaisesRegex(ContractError, "immutable"):
            validate_contract(changed)

    def test_libero_semantic_trees_are_immutable_hashes(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["upstreams"]["libero_official"]["bddl_tree_sha"] = "main"
        with self.assertRaisesRegex(ContractError, "immutable"):
            validate_contract(changed)

    def test_held_writer_inputs_exclude_privileged_fields(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["held_contract"]["writer_visible"].append("reward")
        with self.assertRaisesRegex(ContractError, "Writer-visible"):
            validate_contract(changed)

    def test_dataset_manifest_matches_locked_libero_90_surface(self) -> None:
        dataset = self.contract["datasets"]["libero_90"]
        self.assertEqual(dataset["task_count"], 90)
        self.assertEqual(dataset["file_count"], 90)
        self.assertEqual(dataset["total_bytes"], 66_658_085_995)
        self.assertEqual(dataset["demos_per_task"], 50)
        self.assertEqual(dataset["hdf5_tag"], "libero-v1")
        self.assertEqual(dataset["license"], "CC-BY-4.0")

    def test_official_finetuned_checkpoint_is_mechanics_only(self) -> None:
        smoke = self.contract["models"]["smolvla_libero_smoke"]
        self.assertEqual(smoke["role"], "official_mechanics_only_never_ember_shared_base")
        self.assertNotEqual(
            smoke["revision"], self.contract["models"]["smolvla_base"]["revision"]
        )

    def test_asset_snapshot_bytes_are_not_hub_storage_quota(self) -> None:
        assets = self.contract["datasets"]["libero_assets"]
        self.assertEqual(assets["snapshot_file_count"], 586)
        self.assertEqual(assets["snapshot_bytes"], 422_320_936)
        self.assertEqual(assets["hub_reported_used_storage_bytes"], 492_798_408)
        self.assertNotEqual(
            assets["snapshot_bytes"], assets["hub_reported_used_storage_bytes"]
        )

    def test_video_decoder_backend_is_explicit_and_loadable(self) -> None:
        environment = self.contract["environment"]
        self.assertEqual(environment["video_decode_backend"], "pyav")
        self.assertEqual(environment["pyav"], "15.1.0")

        changed = copy.deepcopy(self.contract)
        changed["environment"]["video_decode_backend"] = "torchcodec"
        with self.assertRaisesRegex(ContractError, "video decoder"):
            validate_contract(changed)


class ActiveResearchScopeTest(unittest.TestCase):
    def test_historical_expert_plan_is_preserved(self) -> None:
        digest = hashlib.sha256((ROOT / "docs" / "expert_plan.md").read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "88f267cc4bfcd5dfdcddb1bc8c945c7191f093fac040292c907fa8676fd5a776",
        )

    def test_active_plan_has_no_bank_or_geometry_milestone(self) -> None:
        plan = (ROOT / "task_plan.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        brief = (ROOT / "docs" / "execution_brief.md").read_text(encoding="utf-8")

        for stale in (
            "| Gate 1. Canonical representation |",
            "| Stage 4. Predicted soft geometry |",
            "Train the geometry before evaluating it",
            "## 5. Corrected geometry training path",
        ):
            self.assertNotIn(stale, plan + agents + brief)
        self.assertIn("Writer emits no second object that constrains RL", brief)
        self.assertIn("shared base policy remains frozen", brief)
        self.assertIn("outside the current EMBER", agents)

    def test_legacy_phase0_geometry_field_is_explicitly_non_authoritative(self) -> None:
        phase0 = (ROOT / "configs" / "phase0.toml").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn('task_geometry = "requires_writer_center_to_rl_benefit"', phase0)
        self.assertIn("non-authoritative/non-executable", agents)

    def test_active_contract_fixes_lora_causal_arms_and_engineering_stop_rule(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        plan = (ROOT / "task_plan.md").read_text(encoding="utf-8")
        brief = (ROOT / "docs" / "execution_brief.md").read_text(encoding="utf-8")
        active = agents + plan + brief

        self.assertIn("LoRA is the only adaptation mechanism", active)
        self.assertIn("LoRA capacity audit", active)
        self.assertIn("zero-LoRA initialization plus ordinary RL", active)
        self.assertIn("cold-start Writer LoRA initialization plus identical RL", active)
        self.assertIn("reward-outer-trained Writer LoRA initialization plus identical RL", active)
        self.assertIn("matched-initial-performance", active)
        self.assertIn("one reproduction, one narrow repair, and one verification", active)
        self.assertIn("OpenVLA-OFT scale confirmation", active)


if __name__ == "__main__":
    unittest.main()
