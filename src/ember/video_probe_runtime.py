"""Pinned data, model, and GPU runtime for the RGB information probe."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ember.contracts import load_contract, validate_contract
from ember.counterfactual_goal_probe import (
    load_probe_spec,
    validate_paired_source_authority,
)
from ember.specification_probe import sha256_file
from ember.video_probe_core import (
    VideoInformationProbeError,
    derive_clip_condition,
    temporal_moment_descriptor,
    uniform_frame_indices,
)


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    prefix = f"{array.dtype.str}:{array.shape}:".encode("ascii")
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


class GpuSampler:
    """Sample physical-device telemetry without shell traps or resident helpers."""

    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "-i",
            str(self.physical_gpu),
            "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) != 4:
            raise VideoInformationProbeError("unexpected nvidia-smi telemetry row")
        self.samples.append(
            {
                "monotonic_s": time.monotonic(),
                "uuid": fields[0],
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop_event.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        try:
            self._sample()
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        if not self.samples:
            raise VideoInformationProbeError("GPU telemetry produced no samples")
        uuids = {row["uuid"] for row in self.samples}
        totals = {row["memory_total_mib"] for row in self.samples}
        if len(uuids) != 1 or len(totals) != 1:
            raise VideoInformationProbeError("GPU telemetry identity changed during run")
        return {
            "physical_gpu": self.physical_gpu,
            "uuid": next(iter(uuids)),
            "sample_count": len(self.samples),
            "peak_total_memory_used_mib": max(row["memory_used_mib"] for row in self.samples),
            "peak_utilization_percent": max(row["utilization_percent"] for row in self.samples),
            "memory_total_mib": next(iter(totals)),
            "sampling_errors": self.errors,
        }


def _validate_model_authority(
    spec: dict[str, Any], contract: dict[str, Any], model_path: Path
) -> dict[str, Any]:
    model = contract["models"][spec["encoder"]["contract_key"]]
    for field in ("repo_id", "revision", "weight_sha256", "weight_bytes"):
        if model[field] != spec["encoder"][field]:
            raise VideoInformationProbeError(f"model contract differs: {field}")
    weight_path = model_path / "model.safetensors"
    if model_path.name != model["revision"] or not weight_path.is_file():
        raise VideoInformationProbeError("pinned SmolVLM snapshot path is incomplete")
    if weight_path.stat().st_size != model["weight_bytes"]:
        raise VideoInformationProbeError("SmolVLM weight size changed")
    if sha256_file(weight_path) != model["weight_sha256"]:
        raise VideoInformationProbeError("SmolVLM weight hash changed")
    return model


def _validate_dataset_authority(
    pair: dict[str, Any], contract: dict[str, Any], dataset_root: Path
) -> dict[str, Any]:
    dataset = contract["datasets"]["libero_90"]
    if dataset_root.name != dataset["subdir"] or dataset_root.parent.name != dataset["revision"]:
        raise VideoInformationProbeError("canonical LIBERO-90 dataset root changed")
    for task in pair["tasks"]:
        path = dataset_root / task["hdf5_filename"]
        if not path.is_file() or path.stat().st_size != task["hdf5_bytes"]:
            raise VideoInformationProbeError("source HDF5 file authority changed")
        if sha256_file(path) != task["hdf5_sha256"]:
            raise VideoInformationProbeError("source HDF5 hash changed")
    return dataset


def load_authority(
    spec: dict[str, Any],
    *,
    source_pair_config: Path,
    contract_path: Path,
    seal_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(source_pair_config) != spec["source_pair_config_sha256"]:
        raise VideoInformationProbeError("source pair config hash changed")
    pair = load_probe_spec(source_pair_config)
    for field in ("task_suite", "task_ids", "source_task_ids", "scene_id", "split_seal_sha256"):
        if pair[field] != spec[field]:
            raise VideoInformationProbeError(f"video/source-pair authority differs: {field}")
    contract = load_contract(contract_path)
    validate_contract(contract)
    manifest_authority, _ = validate_paired_source_authority(
        pair, manifest_path, seal_path, contract
    )
    model = _validate_model_authority(spec, contract, model_path)
    dataset = _validate_dataset_authority(pair, contract, dataset_root)
    authority = {
        **manifest_authority,
        "contract_sha256": sha256_file(contract_path),
        "source_pair_config_sha256": spec["source_pair_config_sha256"],
        "model_revision": model["revision"],
        "model_weight_sha256": model["weight_sha256"],
        "dataset_revision": dataset["revision"],
        "numeric_access": {"source_rgb_arrays": True, "validation": 0, "held_out": 0},
    }
    return pair, authority


def _extract_demo(camera: Any, spec: dict[str, Any]) -> tuple[np.ndarray, ...]:
    expected = (spec["input_height"], spec["input_width"], spec["input_channels"])
    if camera.ndim != 4 or camera.shape[1:] != expected or camera.dtype != np.uint8:
        raise VideoInformationProbeError("source RGB shape or dtype changed")
    full_idx = uniform_frame_indices(
        len(camera),
        frame_count=spec["frame_count"],
        end_exclusion=spec["end_exclusion_frames"],
    )
    short_idx = uniform_frame_indices(
        len(camera),
        frame_count=spec["frame_count"],
        end_exclusion=spec["end_exclusion_frames"],
        end_fraction=0.8,
    )
    union = np.unique(np.concatenate([full_idx, short_idx]))
    frames = np.asarray(camera[union], dtype=np.uint8)
    lookup = {int(value): offset for offset, value in enumerate(union)}
    full = np.stack([frames[lookup[int(value)]] for value in full_idx])
    short = np.stack([frames[lookup[int(value)]] for value in short_idx])
    return full, short, full_idx, short_idx


def extract_clips(
    spec: dict[str, Any], pair: dict[str, Any], dataset_root: Path
) -> dict[str, Any]:
    import h5py

    selected = spec["support_demo_indices"] + spec["query_demo_indices"]
    records, full_clips, short_clips, full_indices, short_indices = [], [], [], [], []
    for task in pair["tasks"]:
        task_id = int(task["task_id"])
        with h5py.File(dataset_root / task["hdf5_filename"], "r") as handle:
            if int(handle["data"].attrs.get("num_demos", -1)) != 50:
                raise VideoInformationProbeError("source HDF5 demonstration count changed")
            for demo_index in selected:
                group = handle["data"].get(f"demo_{demo_index}")
                if group is None or spec["camera_dataset"] not in group:
                    raise VideoInformationProbeError("source RGB demonstration is missing")
                full, short, full_idx, short_idx = _extract_demo(
                    group[spec["camera_dataset"]], spec
                )
                partition = "support" if demo_index in spec["support_demo_indices"] else "query"
                records.append(
                    {
                        "task_id": task_id,
                        "demo_index": demo_index,
                        "partition": partition,
                        "source_frame_count": int(len(group[spec["camera_dataset"]])),
                        "ordered_clip_sha256": _array_sha256(full),
                        "drop_last_20_clip_sha256": _array_sha256(short),
                    }
                )
                full_clips.append(full)
                short_clips.append(short)
                full_indices.append(full_idx)
                short_indices.append(short_idx)
    return {
        "records": records,
        "ordered_clips": np.stack(full_clips),
        "drop_clips": np.stack(short_clips),
        "ordered_indices": np.stack(full_indices),
        "drop_indices": np.stack(short_indices),
    }


def save_clip_cache(path: Path, clips: dict[str, Any]) -> None:
    records = clips["records"]
    np.savez_compressed(
        path,
        ordered_clips=clips["ordered_clips"],
        drop_last_20_clips=clips["drop_clips"],
        ordered_frame_indices=clips["ordered_indices"],
        drop_last_20_frame_indices=clips["drop_indices"],
        task_ids=np.asarray([row["task_id"] for row in records], dtype=np.int16),
        demo_indices=np.asarray([row["demo_index"] for row in records], dtype=np.int16),
        partitions=np.asarray([row["partition"] for row in records]),
        source_frame_counts=np.asarray([row["source_frame_count"] for row in records], dtype=np.int32),
    )


def load_cached_clips(
    spec: dict[str, Any], cache_path: Path, prior_result_path: Path
) -> dict[str, Any]:
    """Validate and reuse the exact RGB cache from the preserved failed run."""

    prior = spec["recovery"]["prior_evidence"]
    if sha256_file(prior_result_path) != prior["probe_result_sha256"]:
        raise VideoInformationProbeError("prior video result hash changed")
    if sha256_file(cache_path) != prior["clip_cache_sha256"]:
        raise VideoInformationProbeError("prior RGB clip cache hash changed")
    result = json.loads(prior_result_path.read_text(encoding="utf-8"))
    if result.get("status") != prior["status"]:
        raise VideoInformationProbeError("prior video failure status changed")
    if result.get("config_sha256") != spec["recovery"]["base_config_sha256"]:
        raise VideoInformationProbeError("prior base video config authority changed")
    if result.get("clip_cache", {}).get("sha256") != prior["clip_cache_sha256"]:
        raise VideoInformationProbeError("prior result does not bind the RGB cache")
    root = prior_result_path.parent
    if sha256_file(root / "feature_cache.npz") != prior["feature_cache_sha256"]:
        raise VideoInformationProbeError("prior failed feature cache hash changed")
    if sha256_file(root / "gallery_manifest.json") != prior["gallery_manifest_sha256"]:
        raise VideoInformationProbeError("prior gallery authority changed")
    with np.load(cache_path, allow_pickle=False) as cached:
        expected_keys = {
            "ordered_clips",
            "drop_last_20_clips",
            "ordered_frame_indices",
            "drop_last_20_frame_indices",
            "task_ids",
            "demo_indices",
            "partitions",
            "source_frame_counts",
        }
        if set(cached.files) != expected_keys:
            raise VideoInformationProbeError("prior RGB clip cache schema changed")
        values = {key: cached[key] for key in cached.files}
    records = result["clip_cache"]["records"]
    expected_clip_shape = (
        96,
        spec["frame_count"],
        spec["input_height"],
        spec["input_width"],
        spec["input_channels"],
    )
    if values["ordered_clips"].shape != expected_clip_shape or values["ordered_clips"].dtype != np.uint8:
        raise VideoInformationProbeError("prior ordered RGB cache shape changed")
    if values["drop_last_20_clips"].shape != expected_clip_shape or len(records) != 96:
        raise VideoInformationProbeError("prior drop-last RGB cache shape changed")
    expected_rows = [
        (row["task_id"], row["demo_index"], row["partition"], row["source_frame_count"])
        for row in records
    ]
    cached_rows = list(
        zip(
            values["task_ids"].tolist(),
            values["demo_indices"].tolist(),
            values["partitions"].tolist(),
            values["source_frame_counts"].tolist(),
        )
    )
    if cached_rows != expected_rows:
        raise VideoInformationProbeError("prior RGB cache row identity changed")
    for index, row in enumerate(records):
        if _array_sha256(values["ordered_clips"][index]) != row["ordered_clip_sha256"]:
            raise VideoInformationProbeError("prior ordered RGB clip digest changed")
        if _array_sha256(values["drop_last_20_clips"][index]) != row["drop_last_20_clip_sha256"]:
            raise VideoInformationProbeError("prior drop-last RGB clip digest changed")
    return {
        "records": records,
        "ordered_clips": values["ordered_clips"],
        "drop_clips": values["drop_last_20_clips"],
        "ordered_indices": values["ordered_frame_indices"],
        "drop_indices": values["drop_last_20_frame_indices"],
    }


def _shuffle_seed(base_seed: int, task_id: int, demo_index: int) -> int:
    payload = f"{base_seed}:{task_id}:{demo_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def feature_descriptors(spec: dict[str, Any], clips: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = [
        {**record, "clip_index": index, "condition": "ordered_full"}
        for index, record in enumerate(clips["records"])
        if record["partition"] == "support"
    ]
    descriptors.extend(
        {**record, "clip_index": index, "condition": condition}
        for condition in spec["conditions"]
        for index, record in enumerate(clips["records"])
        if record["partition"] == "query"
    )
    expected = 2 * 24 + len(spec["conditions"]) * 2 * 24
    if len(descriptors) != expected or len(descriptors) % spec["resources"]["batch_size"]:
        raise VideoInformationProbeError("feature schedule no longer matches the frozen batch")
    return descriptors


def descriptor_clip(
    spec: dict[str, Any], clips: dict[str, Any], row: dict[str, Any]
) -> np.ndarray:
    index = row["clip_index"]
    return derive_clip_condition(
        clips["ordered_clips"][index],
        row["condition"],
        shuffle_seed=_shuffle_seed(spec["shuffle_seed"], row["task_id"], row["demo_index"]),
        drop_last_clip=clips["drop_clips"][index],
    )


def _load_encoder(spec: dict[str, Any], model_path: Path) -> tuple[Any, ...]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise VideoInformationProbeError("probe requires exactly one visible CUDA device")
    torch.manual_seed(spec["shuffle_seed"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).eval().cuda()
    prompt = None
    if spec["encoder"]["feature"] == "final_nonpadding_causal_context_hidden_state":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": spec["encoder"]["prompt"]},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return torch, processor, model, prompt


def _prepare_inputs(
    spec: dict[str, Any], processor: Any, prompt: str, videos: list[np.ndarray]
) -> Any:
    from transformers.video_utils import VideoMetadata

    metadata = [
        VideoMetadata(
            total_num_frames=spec["frame_count"],
            fps=float(spec["standardized_video_fps"]),
            width=spec["input_width"],
            height=spec["input_height"],
            duration=float(spec["frame_count"]),
            frames_indices=list(range(spec["frame_count"])),
        )
        for _ in videos
    ]
    if spec["encoder"]["feature"] == "final_nonpadding_causal_context_hidden_state":
        inputs = processor(
            text=[prompt] * len(videos),
            videos=[[video] for video in videos],
            video_metadata=metadata,
            return_tensors="pt",
            padding=True,
            device="cuda",
        ).to("cuda")
        if not bool(inputs.attention_mask.all()):
            raise VideoInformationProbeError("fixed video batch unexpectedly contains text padding")
    else:
        inputs = processor.video_processor(
            videos,
            video_metadata=metadata,
            return_tensors="pt",
            device="cuda",
        ).to("cuda")
    return inputs


def _forward_batch(
    spec: dict[str, Any], torch: Any, model: Any, inputs: Any
) -> np.ndarray:
    with torch.inference_mode():
        if spec["encoder"]["feature"] == "final_nonpadding_causal_context_hidden_state":
            hidden = model.model(**inputs, return_dict=True).last_hidden_state[:, -1, :]
            batch = hidden.float().cpu().numpy()
        else:
            pixels = inputs.pixel_values
            masks = inputs.pixel_attention_mask
            batch_size, frame_count = pixels.shape[:2]
            visual = model.model.get_image_features(
                pixels, masks, return_dict=True
            ).pooler_output
            frames = visual.mean(dim=1).reshape(batch_size, frame_count, -1)
            batch = temporal_moment_descriptor(frames.float().cpu().numpy())
    torch.cuda.synchronize()
    if not np.isfinite(batch).all():
        raise VideoInformationProbeError("frozen encoder produced a non-finite feature")
    return batch


def encode_features(
    spec: dict[str, Any],
    clips: dict[str, Any],
    descriptors: list[dict[str, Any]],
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch, processor, model, prompt = _load_encoder(spec, model_path)
    batch_size = spec["resources"]["batch_size"]
    features, batch_times = [], []
    repeat_delta: float | None = None
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(descriptors), batch_size):
        rows = descriptors[start : start + batch_size]
        videos = [descriptor_clip(spec, clips, row) for row in rows]
        started = time.perf_counter()
        inputs = _prepare_inputs(spec, processor, prompt, videos)
        batch = _forward_batch(spec, torch, model, inputs)
        batch_times.append(time.perf_counter() - started)
        if start == 0:
            repeated = _forward_batch(spec, torch, model, inputs)
            repeat_delta = float(np.max(np.abs(batch - repeated)))
            if repeat_delta > spec["encoder"]["same_batch_repeat_atol"]:
                raise VideoInformationProbeError("same-batch frozen encoder repeat is unstable")
        features.append(batch)
        del inputs, batch
    result = np.concatenate(features).astype(np.float32)
    telemetry = {
        "feature_shape": list(result.shape),
        "batch_size": batch_size,
        "batch_count": len(batch_times),
        "batch_wall_seconds": batch_times,
        "encoder_wall_seconds": float(sum(batch_times)),
        "clips_per_second": float(len(result) / sum(batch_times)),
        "same_batch_repeat_max_abs_delta": repeat_delta,
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    del model, processor
    torch.cuda.empty_cache()
    return result, telemetry
