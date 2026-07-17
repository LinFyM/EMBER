"""Build safe, local HTML galleries for retained evaluation videos."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any


class EvalArtifactError(RuntimeError):
    """Raised when evaluation outputs are incomplete or unsafe to publish."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_video(run_dir: Path, raw_path: str) -> tuple[Path, Path]:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(run_dir)
    except ValueError as error:
        raise EvalArtifactError(
            f"Evaluation video is outside run directory: {candidate}"
        ) from error
    if not resolved.is_file():
        raise EvalArtifactError(f"Evaluation video is missing: {resolved}")
    return resolved, relative


def _task_cards(
    run_dir: Path, eval_info: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    cards: list[str] = []
    videos: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for task in eval_info.get("per_task", []):
        group = str(task.get("task_group", "unknown"))
        task_id = task.get("task_id", "unknown")
        metrics = task.get("metrics", {})
        successes = [bool(value) for value in metrics.get("successes", [])]
        success_count = sum(successes)
        episode_count = len(successes)
        video_tags: list[str] = []
        for raw_path in metrics.get("video_paths", []):
            resolved, relative = _relative_video(run_dir, str(raw_path))
            if resolved not in seen:
                videos.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": resolved.stat().st_size,
                        "sha256": _sha256(resolved),
                        "task_group": group,
                        "task_id": task_id,
                    }
                )
                seen.add(resolved)
            source = html.escape(relative.as_posix(), quote=True)
            video_tags.append(
                f'<video controls preload="metadata" src="{source}"></video>'
            )
        label = html.escape(f"{group} / task {task_id}")
        summary = html.escape(
            f"success {success_count}/{episode_count}; "
            f"rewards {metrics.get('sum_rewards', [])}"
        )
        cards.append(
            f"<article><h2>{label}</h2><p>{summary}</p>"
            f"{''.join(video_tags) or '<p>No rendered video.</p>'}</article>"
        )
    return cards, videos


def build_eval_gallery(run_dir: Path) -> dict[str, Any]:
    """Create a browser-viewable index and provenance manifest for one run."""

    run_dir = run_dir.resolve()
    info_path = run_dir / "eval_info.json"
    if not info_path.is_file():
        raise EvalArtifactError(f"Evaluation summary is missing: {info_path}")
    eval_info = json.loads(info_path.read_text(encoding="utf-8"))
    cards, videos = _task_cards(run_dir, eval_info)
    overall = html.escape(json.dumps(eval_info.get("overall", {}), sort_keys=True))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EMBER evaluation gallery</title>
  <style>
    body {{ font: 16px system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }}
    article {{ margin: 1.5rem 0; padding: 1rem; background: #1d1d1d; border-radius: .6rem; }}
    video {{ width: min(720px, 100%); margin: .5rem .5rem .5rem 0; background: #000; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>EMBER evaluation gallery</h1>
  <p><code>{overall}</code></p>
  {''.join(cards)}
</body>
</html>
"""
    _atomic_write_text(run_dir / "index.html", document)
    manifest = {
        "schema_version": 1,
        "eval_info_sha256": _sha256(info_path),
        "index_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "videos": videos,
    }
    _atomic_write_text(
        run_dir / "gallery_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {"run_dir": str(run_dir), "video_count": len(videos)}


def update_latest_link(run_dir: Path, latest_link: Path) -> None:
    """Atomically point a symlink at the latest completed run."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise EvalArtifactError(f"Run directory is missing: {run_dir}")
    latest_link = latest_link.absolute()
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(latest_link) and not latest_link.is_symlink():
        raise EvalArtifactError(f"Refusing to replace non-symlink latest path: {latest_link}")
    temporary = latest_link.with_name(f".{latest_link.name}.tmp-{os.getpid()}")
    if os.path.lexists(temporary):
        raise EvalArtifactError(f"Temporary latest link already exists: {temporary}")
    try:
        temporary.symlink_to(
            os.path.relpath(run_dir, latest_link.parent), target_is_directory=True
        )
        os.replace(temporary, latest_link)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--latest-link", type=Path)
    args = parser.parse_args()
    result = build_eval_gallery(args.run_dir)
    if args.latest_link is not None:
        update_latest_link(args.run_dir, args.latest_link)
        result["latest_link"] = str(args.latest_link.absolute())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
