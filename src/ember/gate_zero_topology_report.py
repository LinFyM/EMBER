"""Validate matched 1/2/4-GPU probes and freeze the engineering topology choice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import traceback
from pathlib import Path
from typing import Any, Sequence

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_distributed import (
    GateZeroDistributedError,
    load_distributed_topology_spec,
    select_topology_candidates,
)


class GateZeroTopologyReportError(RuntimeError):
    """Raised when topology evidence is incomplete or incomparable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _numeric_cell(row: dict[str, str], prefix: str) -> float:
    key = next((name for name in row if name.strip().startswith(prefix)), None)
    if key is None:
        raise GateZeroTopologyReportError(f"telemetry lacks {prefix}")
    try:
        return float(row[key].strip().split()[0])
    except (ValueError, IndexError) as error:
        raise GateZeroTopologyReportError(f"invalid telemetry {prefix}") from error


def _telemetry_summary(path: Path, *, expected_world_size: int) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise GateZeroTopologyReportError("cannot read GPU telemetry") from error
    if not rows:
        raise GateZeroTopologyReportError("GPU telemetry is empty")
    indices = {int(_numeric_cell(row, "index")) for row in rows}
    if len(indices) != expected_world_size:
        raise GateZeroTopologyReportError("telemetry GPU count differs from topology")
    free = [_numeric_cell(row, "memory.free") for row in rows]
    used = [_numeric_cell(row, "memory.used") for row in rows]
    utilization = [_numeric_cell(row, "utilization.gpu") for row in rows]
    return {
        "sha256": _sha256(path),
        "row_count": len(rows),
        "gpu_count": len(indices),
        "minimum_free_memory_mib": int(min(free)),
        "maximum_used_memory_mib": int(max(used)),
        "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
    }


def _load_probe(
    probe_dir: Path, *, expected_config_sha256: str
) -> dict[str, Any]:
    result_path = probe_dir / "topology_probe_result.json"
    checksum_path = probe_dir / "checksums.sha256"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroTopologyReportError("topology probe result is missing or invalid") from error
    expected_line = f"{_sha256(result_path)}  topology_probe_result.json"
    if expected_line not in checksum_lines:
        raise GateZeroTopologyReportError("topology probe result checksum changed")
    if result.get("topology_config_sha256") != expected_config_sha256:
        raise GateZeroTopologyReportError("topology probe used a different execution contract")
    for key in (
        "scientific_outcome_metrics_recorded",
        "source_policy_outcome_recorded",
        "gate_zero_authorized",
        "writer_authorized",
    ):
        if result.get(key) is not False:
            raise GateZeroTopologyReportError(f"topology probe violates outcome boundary: {key}")
    telemetry = sorted(probe_dir.glob("gpu_telemetry_*.csv"))
    if len(telemetry) != 1:
        raise GateZeroTopologyReportError("topology probe must have exactly one telemetry file")
    telemetry_checksum_line = f"{_sha256(telemetry[0])}  {telemetry[0].name}"
    if telemetry_checksum_line not in checksum_lines:
        raise GateZeroTopologyReportError("topology probe telemetry checksum changed")
    world_size = result.get("topology", {}).get("world_size")
    summary = _telemetry_summary(telemetry[0], expected_world_size=world_size)
    result["measurement"] = {**result["measurement"], **summary}
    result["probe_basename"] = probe_dir.name
    return result


def _html_report(report: dict[str, Any]) -> str:
    rows = []
    for world_size in (1, 2, 4):
        item = report["candidates"][str(world_size)]
        rows.append(
            "<tr>"
            f"<td>{world_size}</td>"
            f"<td>{item['global_effective_samples_per_second']:.2f}</td>"
            f"<td>{item['speedup_over_single_gpu']:.2f}x</td>"
            f"<td>{item['parallel_efficiency']:.3f}</td>"
            f"<td>{item['minimum_free_memory_mib']}</td>"
            f"<td>{'yes' if item['eligible_for_single_job'] else 'no'}</td>"
            "</tr>"
        )
    summary = html.escape(
        f"selected world size {report['selected_world_size']} via {report['execution_mode']}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMBER Gate 0 topology selection</title>
<style>body{{font:16px system-ui,sans-serif;margin:2rem;background:#111;color:#eee}}table{{border-collapse:collapse}}th,td{{padding:.6rem;border:1px solid #555}}th{{background:#222}}</style>
</head><body><h1>EMBER Gate 0 topology selection</h1><p>{summary}</p>
<table><thead><tr><th>GPUs</th><th>samples/s</th><th>speedup</th><th>efficiency</th><th>min free MiB</th><th>eligible</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Scientific Gate and Writer authorization remain false.</p></body></html>
"""


def build_topology_report(
    *,
    config_path: Path,
    gate_zero_path: Path,
    phase0_path: Path,
    probe_dirs: Sequence[Path],
    output_dir: Path,
    latest_link: Path | None,
) -> dict[str, Any]:
    spec = load_distributed_topology_spec(config_path, gate_zero_path, phase0_path)
    if len(probe_dirs) != 3 or len({path.resolve() for path in probe_dirs}) != 3:
        raise GateZeroTopologyReportError("exactly three distinct topology probes are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise GateZeroTopologyReportError("refusing non-empty topology report directory")
    config_sha256 = _sha256(config_path)
    probes = [
        _load_probe(path.resolve(), expected_config_sha256=config_sha256)
        for path in probe_dirs
    ]
    try:
        selection = select_topology_candidates(spec, probes)
    except GateZeroDistributedError as error:
        raise GateZeroTopologyReportError(str(error)) from error
    report = {
        "schema_version": 1,
        **selection,
        "topology_config_sha256": config_sha256,
        "probe_evidence": {
            str(result["topology"]["world_size"]): {
                "probe_basename": result["probe_basename"],
                "result_sha256": _sha256(
                    next(
                        path / "topology_probe_result.json"
                        for path in probe_dirs
                        if path.name == result["probe_basename"]
                    )
                ),
                "telemetry_sha256": result["measurement"]["sha256"],
            }
            for result in probes
        },
        "topology_contract_amendment_authorized": True,
        "formal_fit_requires_resealed_config_and_same_topology_resume": True,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }
    report_path = output_dir / "topology_selection_report.json"
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    index_path = output_dir / "index.html"
    _atomic_text(index_path, _html_report(report))
    checksum_lines = [
        f"{_sha256(path)}  {path.name}"
        for path in (index_path, report_path)
    ]
    _atomic_text(output_dir / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-zero-contract", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = build_topology_report(
            config_path=args.config.resolve(),
            gate_zero_path=args.gate_zero_contract.resolve(),
            phase0_path=args.phase0_contract.resolve(),
            probe_dirs=args.probe_dir,
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link,
        )
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            args.output_dir / "failure_packet.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
