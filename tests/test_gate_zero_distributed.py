from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_distributed import (  # noqa: E402
    GateZeroDistributedError,
    assert_same_topology,
    global_effective_slots,
    load_distributed_topology_spec,
    merge_rank_provenance,
    require_topology_mode_authorization,
    select_topology_candidates,
    topology_for_world_size,
    validate_distributed_topology_spec,
)


class GateZeroDistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "configs" / "gate_zero_training_topology.toml"
        cls.gate_zero = ROOT / "configs" / "gate_zero_oracle_pilot.toml"
        cls.phase0 = ROOT / "configs" / "phase0.toml"
        cls.spec = load_distributed_topology_spec(cls.path, cls.gate_zero, cls.phase0)

    def test_checked_in_topologies_preserve_global_batch_and_worker_budget(self) -> None:
        for world_size, expected_micro, expected_workers in (
            (1, 64, 4),
            (2, 32, 2),
            (4, 16, 1),
        ):
            topology = topology_for_world_size(self.spec, world_size)
            self.assertEqual(topology.global_effective_batch_size, 64)
            self.assertEqual(topology.per_rank_micro_batch_size, expected_micro)
            self.assertEqual(topology.gradient_accumulation_steps, 1)
            self.assertEqual(topology.data_workers_per_rank, expected_workers)
            self.assertEqual(
                topology.per_rank_micro_batch_size
                * topology.gradient_accumulation_steps
                * topology.world_size,
                64,
            )
            self.assertEqual(topology.data_workers_per_rank * topology.world_size, 4)

    def test_rank_shards_cover_every_global_slot_once(self) -> None:
        for world_size in (1, 2, 4):
            topology = topology_for_world_size(self.spec, world_size)
            slots = [
                slot
                for accumulation_step in range(topology.gradient_accumulation_steps)
                for rank in range(world_size)
                for slot in global_effective_slots(
                    topology,
                    rank=rank,
                    accumulation_step=accumulation_step,
                )
            ]
            self.assertEqual(slots, list(range(64)))
            self.assertEqual(len(slots), len(set(slots)))

    def test_provenance_merge_reconstructs_single_gpu_order_and_digest(self) -> None:
        reference = [f"slot-{slot}" for slot in range(64)]
        for world_size in (1, 2, 4):
            topology = topology_for_world_size(self.spec, world_size)
            per_rank = []
            for rank in range(world_size):
                slots = global_effective_slots(topology, rank=rank, accumulation_step=0)
                per_rank.append([reference[slot] for slot in slots])

            merged = merge_rank_provenance(topology, per_rank)

            self.assertEqual(merged["keys"], reference)
            self.assertEqual(merged["global_slot_count"], 64)
            self.assertEqual(merged["unique_global_slot_count"], 64)
            self.assertEqual(len(merged["sha256"]), 64)

    def test_topology_mismatch_fails_closed_for_resume(self) -> None:
        expected = topology_for_world_size(self.spec, 2).as_manifest()
        assert_same_topology(expected, dict(expected))
        changed = dict(expected)
        changed["world_size"] = 4
        with self.assertRaisesRegex(GateZeroDistributedError, "topology"):
            assert_same_topology(expected, changed)

    def test_contract_rejects_budget_or_authority_drift(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["distributed"]["global_effective_batch_size"] = 128
        with self.assertRaisesRegex(GateZeroDistributedError, "global effective batch"):
            validate_distributed_topology_spec(changed, self.gate_zero, self.phase0)

    def test_pending_probe_allows_mechanics_but_blocks_formal_training(self) -> None:
        require_topology_mode_authorization(
            self.spec, mode="topology-probe", world_size=4
        )
        require_topology_mode_authorization(
            self.spec, mode="resume-probe", world_size=2
        )
        with self.assertRaisesRegex(GateZeroDistributedError, "awaits"):
            require_topology_mode_authorization(
                self.spec, mode="train", world_size=4
            )
        changed = copy.deepcopy(self.spec)
        changed["selection"]["scientific_thresholds_may_change"] = True
        with self.assertRaisesRegex(GateZeroDistributedError, "scientific thresholds"):
            validate_distributed_topology_spec(changed, self.gate_zero, self.phase0)

    def _candidate(self, world_size: int, throughput: float, free_mib: int = 60000):
        return {
            "status": "topology_probe_completed_pending_cross_topology_selection",
            "topology": {"world_size": world_size},
            "measurement": {
                "global_effective_samples_per_second": throughput,
                "minimum_free_memory_mib": free_mib,
            },
            "same_topology_resume": {"all_exact": True},
            "global_authority": {
                "initial_model_state_sha256": "1" * 64,
                "initial_optimizer_state_sha256": "2" * 64,
                "initial_scheduler_state_sha256": "3" * 64,
                "row_keys_sha256_by_step": ["a" * 64, "b" * 64],
                "flow_input_sha256_by_step": ["c" * 64, "d" * 64],
                "global_slot_count_by_step": [64, 64],
                "unique_global_slot_count_by_step": [64, 64],
            },
        }

    def test_selection_chooses_fastest_safe_scaling_or_independent_jobs(self) -> None:
        selected = select_topology_candidates(
            self.spec,
            [self._candidate(1, 90), self._candidate(2, 160), self._candidate(4, 210)],
        )
        self.assertEqual(selected["execution_mode"], "ddp")
        self.assertEqual(selected["selected_world_size"], 4)
        self.assertAlmostEqual(selected["candidates"]["4"]["parallel_efficiency"], 210 / 90 / 4)

        independent = select_topology_candidates(
            self.spec,
            [self._candidate(1, 90), self._candidate(2, 100), self._candidate(4, 120)],
        )
        self.assertEqual(independent["execution_mode"], "independent_job_parallelism")
        self.assertEqual(independent["selected_world_size"], 1)
        self.assertEqual(independent["maximum_concurrent_independent_jobs"], 4)

    def test_selection_rejects_cross_topology_authority_drift(self) -> None:
        candidates = [self._candidate(1, 90), self._candidate(2, 160), self._candidate(4, 210)]
        candidates[2]["global_authority"]["flow_input_sha256_by_step"][1] = "0" * 64
        with self.assertRaisesRegex(GateZeroDistributedError, "flow|authority"):
            select_topology_candidates(self.spec, candidates)

        candidates = [self._candidate(1, 90), self._candidate(2, 160), self._candidate(4, 210)]
        candidates[1]["global_authority"]["initial_model_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(GateZeroDistributedError, "initial|authority"):
            select_topology_candidates(self.spec, candidates)


if __name__ == "__main__":
    unittest.main()
