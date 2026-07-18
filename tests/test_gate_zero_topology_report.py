from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_topology_report import (  # noqa: E402
    GateZeroTopologyReportError,
    build_topology_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GateZeroTopologyReportTest(unittest.TestCase):
    def _probe(self, root: Path, world_size: int, throughput: float, *, drift: bool = False) -> Path:
        probe = root / f"probe_world_{world_size}"
        probe.mkdir()
        flow = ["c" * 64, ("0" if drift else "d") * 64]
        result = {
            "schema_version": 1,
            "status": "topology_probe_completed_pending_cross_topology_selection",
            "scientific_outcome_metrics_recorded": False,
            "source_policy_outcome_recorded": False,
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "topology_config_sha256": _sha256(
                ROOT / "configs" / "gate_zero_training_topology.toml"
            ),
            "topology": {"world_size": world_size},
            "measurement": {"global_effective_samples_per_second": throughput},
            "same_topology_resume": {"all_exact": True},
            "global_authority": {
                "initial_model_state_sha256": "1" * 64,
                "initial_optimizer_state_sha256": "2" * 64,
                "initial_scheduler_state_sha256": "3" * 64,
                "row_keys_sha256_by_step": ["a" * 64, "b" * 64],
                "flow_input_sha256_by_step": flow,
                "global_slot_count_by_step": [64, 64],
                "unique_global_slot_count_by_step": [64, 64],
            },
        }
        result_path = probe / "topology_probe_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        rows = [
            "timestamp, index, uuid, memory.used [MiB], memory.free [MiB], utilization.gpu [%], utilization.memory [%], power.draw [W]"
        ]
        for index in range(world_size):
            rows.append(
                f"2026/07/18 10:00:0{index}.000, {4 + index}, GPU-{index}, 20000 MiB, 61000 MiB, 99 %, 80 %, 300.00 W"
            )
        telemetry_path = probe / "gpu_telemetry_20260718T100000Z.csv"
        telemetry_path.write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        (probe / "checksums.sha256").write_text(
            f"{_sha256(result_path)}  topology_probe_result.json\n"
            f"{_sha256(telemetry_path)}  {telemetry_path.name}\n",
            encoding="utf-8",
        )
        return probe

    def test_report_selects_fastest_safe_topology_and_builds_local_html(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = [
                self._probe(root, 1, 90.0),
                self._probe(root, 2, 160.0),
                self._probe(root, 4, 210.0),
            ]
            output = root / "selection"

            report = build_topology_report(
                config_path=ROOT / "configs" / "gate_zero_training_topology.toml",
                gate_zero_path=ROOT / "configs" / "gate_zero_oracle_pilot.toml",
                phase0_path=ROOT / "configs" / "phase0.toml",
                probe_dirs=probes,
                output_dir=output,
                latest_link=root / "latest",
            )

            self.assertEqual(report["selected_world_size"], 4)
            self.assertEqual(report["execution_mode"], "ddp")
            self.assertTrue(report["topology_contract_amendment_authorized"])
            self.assertFalse(report["gate_zero_authorized"])
            self.assertEqual(report["candidates"]["4"]["minimum_free_memory_mib"], 61000)
            self.assertTrue((output / "index.html").is_file())
            self.assertEqual((root / "latest").resolve(), output.resolve())
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertNotIn(str(root), html)

    def test_report_rejects_cross_topology_flow_authority_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = [
                self._probe(root, 1, 90.0),
                self._probe(root, 2, 160.0),
                self._probe(root, 4, 210.0, drift=True),
            ]
            with self.assertRaisesRegex(GateZeroTopologyReportError, "flow|authority"):
                build_topology_report(
                    config_path=ROOT / "configs" / "gate_zero_training_topology.toml",
                    gate_zero_path=ROOT / "configs" / "gate_zero_oracle_pilot.toml",
                    phase0_path=ROOT / "configs" / "phase0.toml",
                    probe_dirs=probes,
                    output_dir=root / "selection",
                    latest_link=None,
                )

    def test_report_rejects_telemetry_changed_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = [
                self._probe(root, 1, 90.0),
                self._probe(root, 2, 160.0),
                self._probe(root, 4, 210.0),
            ]
            telemetry = next(probes[1].glob("gpu_telemetry_*.csv"))
            telemetry.write_text(
                telemetry.read_text(encoding="utf-8").replace("61000 MiB", "9000 MiB"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(GateZeroTopologyReportError, "telemetry.*checksum"):
                build_topology_report(
                    config_path=ROOT / "configs" / "gate_zero_training_topology.toml",
                    gate_zero_path=ROOT / "configs" / "gate_zero_oracle_pilot.toml",
                    phase0_path=ROOT / "configs" / "phase0.toml",
                    probe_dirs=probes,
                    output_dir=root / "selection",
                    latest_link=None,
                )


if __name__ == "__main__":
    unittest.main()
