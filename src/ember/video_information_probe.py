"""Run the canonical source-only action-hidden RGB information diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ember.eval_artifacts import update_latest_link
from ember.specification_probe import sha256_file
from ember.video_probe_core import (
    VideoInformationProbeError,
    decide_video_probe,
    derive_clip_condition,
    fit_frozen_linear_probe,
    load_video_recovery_spec,
    load_video_spec,
    score_linear_probe,
    stratified_accuracy_interval,
    temporal_moment_descriptor,
    uniform_frame_indices,
)
from ember.video_probe_runtime import (
    GpuSampler,
    descriptor_clip,
    encode_features,
    extract_clips,
    feature_descriptors,
    load_authority,
    load_cached_clips,
    save_clip_cache,
)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _save_feature_cache(
    path: Path, descriptors: list[dict[str, Any]], features: np.ndarray
) -> None:
    np.savez_compressed(
        path,
        features=features,
        partitions=np.asarray([row["partition"] for row in descriptors]),
        conditions=np.asarray([row["condition"] for row in descriptors]),
        task_ids=np.asarray([row["task_id"] for row in descriptors], dtype=np.int16),
        demo_indices=np.asarray([row["demo_index"] for row in descriptors], dtype=np.int16),
    )


def _condition_score(
    spec: dict[str, Any],
    descriptors: list[dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    readout: dict[str, Any],
    condition: str,
    condition_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask = np.asarray(
        [row["partition"] == "query" and row["condition"] == condition for row in descriptors]
    )
    scored = score_linear_probe(readout, features[mask], labels[mask])
    interval = stratified_accuracy_interval(
        scored["correct"],
        labels[mask],
        task_ids=spec["task_ids"],
        samples=spec["readout"]["bootstrap_samples"],
        seed=spec["readout"]["bootstrap_seed"] + condition_index,
        confidence_level=spec["readout"]["confidence_level"],
    )
    metric = {
        "balanced_accuracy": scored["balanced_accuracy"],
        "per_task_accuracy": scored["per_task_accuracy"],
        "bootstrap_interval": interval,
        "mean_signed_margin": float(np.mean(scored["signed_margins"])),
        "minimum_signed_margin": float(np.min(scored["signed_margins"])),
        "query_count": int(mask.sum()),
    }
    selected = [row for row, keep in zip(descriptors, mask) if keep]
    records = [
        {
            "condition": condition,
            "task_id": row["task_id"],
            "demo_index": row["demo_index"],
            "score": float(score),
            "prediction": int(prediction),
            "signed_margin": float(margin),
        }
        for row, score, prediction, margin in zip(
            selected, scored["scores"], scored["predictions"], scored["signed_margins"]
        )
    ]
    return metric, records


def _bidirectional_pair_metrics(
    spec: dict[str, Any], per_query: list[dict[str, Any]]
) -> dict[str, Any]:
    predictions = {
        (row["task_id"], row["demo_index"]): row["prediction"]
        for row in per_query
        if row["condition"] == "ordered_full"
    }
    switches = [
        all(predictions[(task_id, demo)] == task_id for task_id in spec["task_ids"])
        for demo in spec["query_demo_indices"]
    ]
    return {
        "bidirectional_query_pair_fraction": float(np.mean(switches)),
        "bidirectional_query_pair_count": int(sum(switches)),
        "bidirectional_query_pair_total": len(switches),
    }


def _score_conditions(
    spec: dict[str, Any], descriptors: list[dict[str, Any]], features: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray([row["task_id"] for row in descriptors])
    support = np.asarray([row["partition"] == "support" for row in descriptors])
    readout = fit_frozen_linear_probe(
        features[support],
        labels[support],
        task_ids=spec["task_ids"],
        ridge_lambda=spec["readout"]["ridge_lambda"],
    )
    metrics, per_query = {}, []
    for index, condition in enumerate(spec["conditions"]):
        metric, records = _condition_score(
            spec, descriptors, features, labels, readout, condition, index
        )
        metrics[condition] = metric
        per_query.extend(records)
    metrics.update(_bidirectional_pair_metrics(spec, per_query))
    metrics["wrong_video_specificity"] = metrics["ordered_full"]["balanced_accuracy"]
    metrics["wrong_video_mean_correct_minus_swapped_score"] = (
        2 * metrics["ordered_full"]["mean_signed_margin"]
    )
    summary = {"metrics": metrics, "decision": decide_video_probe(spec, metrics)}
    details = {
        "kind": spec["readout"]["kind"],
        "ridge_lambda": readout["ridge_lambda"],
        "weight_norm": readout["weight_norm"],
        "support_count": int(support.sum()),
        "per_query": per_query,
    }
    return summary, details


def _encode_mp4(path: Path, clip: np.ndarray, *, fps: int = 4) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.mp4")
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream(
                "libx264", rate=fps, options={"crf": "23", "preset": "veryfast"}
            )
            stream.width, stream.height, stream.pix_fmt = clip.shape[2], clip.shape[1], "yuv420p"
            for image in clip:
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_gallery(
    output_dir: Path,
    spec: dict[str, Any],
    clips: dict[str, Any],
    summary: dict[str, Any],
    reuse_root: Path | None,
) -> dict[str, Any]:
    cards, videos = [], []
    if reuse_root is not None:
        (output_dir / "videos").symlink_to(
            os.path.relpath(reuse_root / "videos", output_dir), target_is_directory=True
        )
    for task_id in spec["task_ids"]:
        index = next(
            i
            for i, row in enumerate(clips["records"])
            if row["task_id"] == task_id and row["demo_index"] == spec["gallery_demo_index"]
        )
        base = {**clips["records"][index], "clip_index": index}
        for condition in spec["conditions"]:
            clip = descriptor_clip(spec, clips, {**base, "condition": condition})
            relative = Path("videos") / f"task_{task_id}" / f"{condition}.mp4"
            path = output_dir / relative
            if reuse_root is None:
                _encode_mp4(path, clip)
            elif sha256_file(reuse_root / relative) != sha256_file(path):
                raise VideoInformationProbeError("reused gallery video digest changed")
            videos.append(
                {
                    "task_id": task_id,
                    "demo_index": spec["gallery_demo_index"],
                    "condition": condition,
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "reused_from_prior": reuse_root is not None,
                }
            )
            cards.append(
                f"<article><h2>task {task_id} / {html.escape(condition)}</h2>"
                f'<video controls preload="metadata" src="{relative.as_posix()}"></video></article>'
            )
    decision = html.escape(json.dumps(summary["decision"], sort_keys=True))
    metrics = html.escape(json.dumps(summary["metrics"], sort_keys=True))
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EMBER video probe</title>
<style>body{{font:16px system-ui,sans-serif;margin:2rem;background:#111;color:#eee}}article{{display:inline-block;vertical-align:top;margin:.6rem;padding:.8rem;background:#1d1d1d;border-radius:.5rem}}video{{width:384px;max-width:90vw;background:#000}}code{{white-space:pre-wrap}}</style></head>
<body><h1>EMBER source-only action-hidden video probe</h1><p>Two-source-task diagnostic; not a Gate -1 pass or Writer result.</p><h2>Decision</h2><code>{decision}</code><h2>Metrics</h2><code>{metrics}</code>{''.join(cards)}</body></html>"""
    _atomic_text(output_dir / "index.html", document)
    gallery = {
        "schema_version": 1,
        "index_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "media_reused_from_prior": reuse_root is not None,
        "videos": videos,
    }
    _atomic_json(output_dir / "gallery_manifest.json", gallery)
    return gallery


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in paths]
    _atomic_text(output_dir / "checksums.sha256", "\n".join(lines) + "\n")


def _check_gpu_headroom(spec: dict[str, Any], telemetry: dict[str, Any]) -> None:
    minimum = spec["resources"]["minimum_gpu_headroom_gib"] * 1024
    observed = telemetry["memory_total_mib"] - telemetry["peak_total_memory_used_mib"]
    if observed < minimum:
        raise VideoInformationProbeError("measured total GPU headroom fell below contract")


def _encode_with_telemetry(
    spec: dict[str, Any],
    clips: dict[str, Any],
    descriptors: list[dict[str, Any]],
    model_path: Path,
    physical_gpu: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    sampler = GpuSampler(physical_gpu)
    sampler.start()
    try:
        features, encoder = encode_features(spec, clips, descriptors, model_path)
    finally:
        gpu = sampler.stop()
    _check_gpu_headroom(spec, gpu)
    return features, encoder, gpu


def _build_result(
    spec: dict[str, Any],
    authority: dict[str, Any],
    clips: dict[str, Any],
    descriptors: list[dict[str, Any]],
    features: np.ndarray,
    summary: dict[str, Any],
    output_dir: Path,
    gallery: dict[str, Any],
    active_config_path: Path,
    base_config_path: Path,
    clip_cache_info: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "status": summary["decision"]["status"],
        "config_sha256": sha256_file(active_config_path),
        "base_config_sha256": sha256_file(base_config_path),
        "spec": spec,
        "authority": authority,
        "input_contract": {
            "writer_visible": ["16 standardized third-person RGB frames"],
            "not_visible": [
                "task language",
                "task ID",
                "actions",
                "proprioception",
                "reward",
                "terminal flags",
                "source trajectory length",
                "filenames",
                "normalization statistics",
            ],
            "source_only_label_use": spec["encoder"]["source_label_use"],
        },
        "clip_cache": clip_cache_info,
        "feature_cache": {
            "path": "feature_cache.npz",
            "sha256": sha256_file(output_dir / "feature_cache.npz"),
            "record_count": len(descriptors),
            "shape": list(features.shape),
        },
        **summary,
        "gallery": {
            "path": "index.html",
            "video_count": len(gallery["videos"]),
            "manifest_sha256": sha256_file(output_dir / "gallery_manifest.json"),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    if "recovery" in spec:
        result["recovery_boundary"] = spec["recovery"]["claim_boundary"]
    return result


def _load_run_spec(
    base_config_path: Path, recovery_config_path: Path | None
) -> tuple[dict[str, Any], Path]:
    if recovery_config_path is None:
        return load_video_spec(base_config_path), base_config_path
    return (
        load_video_recovery_spec(recovery_config_path, base_config_path),
        recovery_config_path,
    )


def _load_or_extract_clips(
    spec: dict[str, Any],
    pair: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    prior_result_path: Path | None,
    input_clip_cache: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    if "recovery" not in spec:
        if prior_result_path is not None or input_clip_cache is not None:
            raise VideoInformationProbeError("base probe cannot consume prior evidence")
        clips = extract_clips(spec, pair, dataset_root)
        cache_path = output_dir / "clip_cache.npz"
        save_clip_cache(cache_path, clips)
        info = {
            "path": "clip_cache.npz",
            "sha256": sha256_file(cache_path),
            "record_count": len(clips["records"]),
            "records": clips["records"],
            "reused": False,
        }
        return clips, info, None
    if prior_result_path is None or input_clip_cache is None:
        raise VideoInformationProbeError("recovery requires its frozen prior result and RGB cache")
    clips = load_cached_clips(spec, input_clip_cache, prior_result_path)
    info = {
        "path": None,
        "sha256": spec["recovery"]["prior_evidence"]["clip_cache_sha256"],
        "record_count": len(clips["records"]),
        "records": clips["records"],
        "reused": True,
        "prior_result_sha256": spec["recovery"]["prior_evidence"]["probe_result_sha256"],
    }
    return clips, info, prior_result_path.parent


def run_probe(
    *,
    config_path: Path,
    source_pair_config: Path,
    contract_path: Path,
    seal_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    output_dir: Path,
    latest_link: Path | None,
    physical_gpu: int,
    recovery_config_path: Path | None = None,
    prior_result_path: Path | None = None,
    input_clip_cache: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise VideoInformationProbeError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    spec, active_config_path = _load_run_spec(config_path, recovery_config_path)
    pair, authority = load_authority(
        spec,
        source_pair_config=source_pair_config,
        contract_path=contract_path,
        seal_path=seal_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        model_path=model_path,
    )
    extraction_started = time.perf_counter()
    clips, clip_cache_info, reuse_root = _load_or_extract_clips(
        spec,
        pair,
        dataset_root,
        output_dir,
        prior_result_path,
        input_clip_cache,
    )
    extraction_seconds = time.perf_counter() - extraction_started
    descriptors = feature_descriptors(spec, clips)
    features, encoder_telemetry, gpu_telemetry = _encode_with_telemetry(
        spec, clips, descriptors, model_path, physical_gpu
    )
    _save_feature_cache(output_dir / "feature_cache.npz", descriptors, features)
    summary, readout = _score_conditions(spec, descriptors, features)
    _atomic_json(output_dir / "readout_details.json", readout)
    gallery = _build_gallery(output_dir, spec, clips, summary, reuse_root)
    telemetry = {
        "wall_seconds": time.perf_counter() - started,
        "clip_extraction_or_reuse_seconds": extraction_seconds,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "encoder": encoder_telemetry,
        "gpu": gpu_telemetry,
        "artifact_budget_gib": spec["resources"]["expected_output_gib"],
    }
    _atomic_json(output_dir / "resource_telemetry.json", telemetry)
    result = _build_result(
        spec,
        authority,
        clips,
        descriptors,
        features,
        summary,
        output_dir,
        gallery,
        active_config_path,
        config_path,
        clip_cache_info,
    )
    _atomic_json(output_dir / "probe_result.json", result)
    _write_checksums(output_dir)
    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    if total_bytes > spec["resources"]["expected_output_gib"] * 2**30:
        raise VideoInformationProbeError("retained output exceeded the artifact budget")
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    return {
        "output_dir": str(output_dir),
        "latest_link": str(latest_link) if latest_link else None,
        "status": result["status"],
        "artifact_bytes": total_bytes,
        "video_count": len(gallery["videos"]),
        "wall_seconds": telemetry["wall_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-pair-config", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--latest-link", type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--recovery-config", type=Path)
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--input-clip-cache", type=Path)
    args = parser.parse_args()
    try:
        result = run_probe(
            config_path=args.config.resolve(),
            source_pair_config=args.source_pair_config.resolve(),
            contract_path=args.contract.resolve(),
            seal_path=args.seal.resolve(),
            manifest_path=args.manifest.resolve(),
            dataset_root=args.dataset_root.resolve(),
            model_path=args.model_path.resolve(),
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link.absolute() if args.latest_link else None,
            physical_gpu=args.physical_gpu,
            recovery_config_path=(
                args.recovery_config.resolve() if args.recovery_config else None
            ),
            prior_result_path=args.prior_result.resolve() if args.prior_result else None,
            input_clip_cache=(
                args.input_clip_cache.resolve() if args.input_clip_cache else None
            ),
        )
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            args.output_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "gate_decision_authorized": False,
                "writer_authorized": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
