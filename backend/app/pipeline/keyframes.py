"""Keyframe sampling lane — extract a bounded set of downscaled frames.

EFFICIENCY-CRITICAL: we never full-decode the video. A single ffmpeg call with
an ``fps`` filter samples at most ``max_frames`` frames, downscaled so the long
edge is at most ``keyframe_max_dim``. Each JPEG is lazily opened into a
``PIL.Image`` when Pillow is available; otherwise the path is kept and
``image=None``.

Returns ``[]`` when there is no ffmpeg, no video, or any error — the caller
marks ``degradation.media``. Never raises.

Only Python stdlib is imported at module top level.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from typing import List, Optional

from app.config import UPLOAD_DIR, settings
from app.models import registry
from app.pipeline.contracts import Keyframe, MediaInput, ProbeResult

_EXTRACT_TIMEOUT_S = 120


def _input_path(media: MediaInput) -> Optional[str]:
    """Return a readable upload path, or None for non-uploads / missing files."""
    if media.source_type != "upload":
        return None
    path = media.path
    if not path:
        return None
    try:
        if os.path.isfile(path) and os.access(path, os.R_OK):
            return path
    except OSError:
        return None
    return None


def _compute_fps(duration_s: float, interval_s: float, max_frames: int) -> float:
    """Frames-per-second that keeps the total frame count <= ``max_frames``.

    ``fps = min(1/interval_s, max_frames/max(duration_s, 1))`` — the second
    term spreads ``max_frames`` evenly across long clips so we never overshoot.
    """
    by_interval = 1.0 / interval_s if interval_s > 0 else 1.0
    by_budget = max_frames / max(duration_s, 1.0)
    fps = min(by_interval, by_budget)
    # Guard against zero / negative fps for degenerate inputs.
    if fps <= 0:
        fps = by_interval if by_interval > 0 else 1.0
    return fps


def extract_keyframes(
    media: MediaInput,
    probe: ProbeResult,
    interval_s: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[Keyframe]:
    """Sample downscaled keyframes from the upload's video stream.

    One ffmpeg call samples frames at a computed fps; resulting JPEGs are read
    back into :class:`Keyframe` records. Returns ``[]`` on any failure. Never
    raises.
    """
    interval_s = settings.keyframe_interval_s if interval_s is None else interval_s
    max_frames = settings.max_keyframes if max_frames is None else max_frames
    max_dim = settings.keyframe_max_dim

    if max_frames <= 0:
        return []

    path = _input_path(media)
    if path is None:
        return []

    # If we positively know there is no video, there is nothing to sample.
    if probe.ok and not probe.has_video:
        return []

    ffmpeg = registry.ffmpeg_exe()
    if not ffmpeg:
        return []

    fps = _compute_fps(probe.duration_s, interval_s, max_frames)

    out_dir = UPLOAD_DIR / f"keyframes_{uuid.uuid4().hex}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []

    pattern = str(out_dir / "kf_%03d.jpg")
    # fps then scale: long edge capped at max_dim, width kept even (-2).
    vf = f"fps={fps:.6f},scale='min({max_dim},iw)':-2"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-vf", vf,
        "-frames:v", str(max_frames),
        pattern,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_EXTRACT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        # Frames may still have been written before an error; only use them if
        # the run succeeded to avoid surfacing partial garbage.
        return []

    try:
        files = sorted(
            f for f in os.listdir(out_dir)
            if f.startswith("kf_") and f.lower().endswith(".jpg")
        )
    except OSError:
        return []

    keyframes: List[Keyframe] = []
    for index, name in enumerate(files):
        frame_path = str(out_dir / name)
        image = _load_image(frame_path)
        keyframes.append(
            Keyframe(
                index=index,
                time_s=index / fps if fps > 0 else 0.0,
                image=image,
                path=frame_path,
            )
        )
    return keyframes


def _load_image(path: str) -> object:
    """Lazily open a JPEG into a PIL.Image, or return None if Pillow is absent
    or the file cannot be decoded. Never raises."""
    try:
        from PIL import Image  # type: ignore
    except Exception:  # noqa: BLE001 - Pillow missing -> keep path only
        return None
    try:
        img = Image.open(path)
        img.load()  # force decode now so the file handle can close
        return img
    except Exception:  # noqa: BLE001 - corrupt/unreadable frame
        return None
