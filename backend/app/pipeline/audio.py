"""Audio extraction lane — transcode an upload's audio to 16 kHz mono WAV.

The WAV is what the ASR lane (whisper / faster-whisper) consumes. We never load
samples into memory here — only produce a file path. Degrades to ``None`` when
there is no ffmpeg, no audio stream, or any transcode error, so the caller can
mark ``degradation.asr`` and skip ASR.

Only Python stdlib is imported at module top level.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from typing import Optional

from app.config import UPLOAD_DIR
from app.models import registry
from app.pipeline.contracts import AudioTrack, MediaInput, ProbeResult

# Transcoding a short social clip is fast; bound it generously regardless.
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


def extract_audio(media: MediaInput, probe: ProbeResult) -> Optional[AudioTrack]:
    """Extract a 16 kHz mono WAV from the upload's audio track.

    Returns an :class:`AudioTrack` pointing at the written WAV on success, or
    ``None`` when extraction is impossible (no ffmpeg, no audio stream, error).
    Never raises.
    """
    path = _input_path(media)
    if path is None:
        return None

    # If we positively know there is no audio, don't bother spawning ffmpeg.
    if probe.ok and not probe.has_audio:
        return None

    ffmpeg = registry.ffmpeg_exe()
    if not ffmpeg:
        return None

    out_path = UPLOAD_DIR / f"audio_{uuid.uuid4().hex}.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-vn",                 # drop video
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        str(out_path),
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
        _cleanup(out_path)
        return None

    # ffmpeg returns non-zero (and writes no/empty file) when there is no audio
    # stream or the input is unreadable.
    if proc.returncode != 0:
        _cleanup(out_path)
        return None

    try:
        if not out_path.is_file() or out_path.stat().st_size == 0:
            _cleanup(out_path)
            return None
    except OSError:
        _cleanup(out_path)
        return None

    return AudioTrack(
        path=str(out_path),
        sample_rate=16000,
        duration_s=max(0.0, probe.duration_s),
        samples=None,
    )


def _cleanup(path) -> None:
    """Best-effort removal of a partial/empty output file."""
    try:
        os.remove(path)
    except OSError:
        pass
