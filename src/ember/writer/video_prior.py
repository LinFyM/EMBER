"""Frozen, prefix-local V-JEPA evidence; no policy, task identity or action inputs."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor

from ember.writer.native import autocast


PRIOR_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
PRIOR_MODES = ("ordered", "frame_set", "frame_set_image")
PATCH_COUNT = 24 * 24
PRIOR_WIDTH = 1024


def validate_prior_config(config: dict) -> None:
    prior = config["video_prior"]
    mode = prior.get("mode")
    expected_process = "ordered" if mode == "ordered" else "frame_set"
    fixed = {"schema": "frozen_vjepa21_prefix4_v1", "code_commit": PRIOR_COMMIT,
             "checkpoint_key": "ema_encoder", "input_size": 384, "prefix_frames": 4,
             "preprocessing": "official_single_view_resize438_center384_imagenet",
             "code_root": ".codex/vendor/vjepa2",
             "checkpoint": "models/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"}
    if (mode not in PRIOR_MODES or config["model"]["process_mode"] != expected_process
            or config["observer"].get("camera_view", "agentview") != "agentview"
            or {key: prior.get(key) for key in fixed} != fixed
            or type(prior.get("window_batch")) is not int or prior["window_batch"] <= 0
            or set(prior) != {*fixed, "mode", "window_batch"}):
        raise ValueError("frozen video prior scientific or asset contract changed")


def prefix_windows(count: int, mode: str) -> Tensor:
    """Input positions for every actual frame; a static row never reads another frame."""
    if count <= 0 or mode not in PRIOR_MODES:
        raise ValueError("video prior needs real frames and a registered process mode")
    now = torch.arange(count)[:, None]
    if mode == "ordered":
        return (now + torch.arange(-3, 1)[None]).clamp_min(0)
    return now.expand(-1, 1 if mode == "frame_set_image" else 4)


def _official_modules(code_root: Path):
    code_root = code_root.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PRIOR_COMMIT:
        raise ValueError("V-JEPA source differs from the registered frozen commit")
    subprocess.run(["git", "-C", str(code_root), "diff", "--quiet", "HEAD"], check=True)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    model = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    transforms = importlib.import_module("evals.video_classification_frozen.utils")
    for module in (model, transforms):
        if not Path(module.__file__).resolve().is_relative_to(code_root):
            raise ValueError("a different imported V-JEPA source owns this process")
    return model, transforms


class FrozenVideoPrior:
    """Own one frozen encoder outside Writer state/optimizer/checkpoints.

    Returned CPU dense tokens may be reused by a bounded frozen-input cache.
    All learned task grounding remains in the Writer and is replayed normally.
    """

    def __init__(self, code_root: Path, checkpoint: Path, device: torch.device,
                 *, mode: str, window_batch: int) -> None:
        if mode not in PRIOR_MODES or window_batch <= 0:
            raise ValueError("invalid frozen video prior execution configuration")
        model, transforms = _official_modules(code_root)
        # Frozen model construction must not change the fresh Writer RNG stream.
        with torch.random.fork_rng(devices=[]):
            encoder = model.vit_large(
                patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2,
                use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        weights = {key.replace("module.", "").replace("backbone.", ""): value
                   for key, value in payload["ema_encoder"].items()}
        encoder.load_state_dict(weights, strict=True)
        del weights, payload
        self.encoder = encoder.requires_grad_(False).eval().to(device)
        self.transform = transforms.VideoTransform(training=False, crop_size=384)
        self.device, self.mode, self.window_batch = device, mode, int(window_batch)

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        if (frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3
                or not len(frames)):
            raise ValueError("frozen prior requires real uint8 single-camera RGB [T,3,H,W]")
        # Apply the official deterministic spatial transform once per real frame.
        pixels = self.transform(frames.cpu().permute(0, 2, 3, 1).numpy())[0]
        windows = prefix_windows(len(frames), self.mode)
        dense = []
        for start in range(0, len(frames), self.window_batch):
            indices = windows[start:start + self.window_batch]
            clips = pixels[:, indices].permute(1, 0, 2, 3, 4).contiguous().to(self.device)
            with autocast(self.device):
                tokens = self.encoder(clips)
            temporal = 1 if self.mode == "frame_set_image" else 2
            if tokens.shape != (len(indices), temporal * PATCH_COUNT, PRIOR_WIDTH):
                raise ValueError("frozen prior lost its dense final-layer patch layout")
            current = tokens.reshape(len(indices), temporal, PATCH_COUNT, PRIOR_WIDTH)[:, -1]
            if not torch.isfinite(current).all():
                raise ValueError("frozen prior produced nonfinite visual evidence")
            dense.append(current.to(device="cpu", dtype=torch.bfloat16))
        return torch.cat(dense)
