"""Media probing lane — extract container/stream metadata from an uploaded file.

Prefers ``ffprobe`` (clean JSON), falls back to parsing ``ffmpeg`` stderr, and
degrades to ``ProbeResult(ok=False)`` when neither binary is available or the
input is unreadable. Never raises: every failure mode returns a sentinel result
the orchestrator can branch on (it sets ``degradation.media``).

Only Python stdlib is imported at module top level.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Optional

from app.models import registry
from app.pipeline.contracts import MediaInput, ProbeResult

# ffprobe/ffmpeg can hang on broken files; bound every external call.
_PROBE_TIMEOUT_S = 30

# Matches e.g. "Duration: 00:01:23.45," in ffmpeg's stderr banner.
_DURATION_RX = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", re.IGNORECASE
)
# Matches a video stream line: "Stream #0:0 ... Video: h264 ... 1280x720 ..."
_VIDEO_RX = re.compile(r"Stream #\d+:\d+.*: Video:.*?(\d{2,5})x(\d{2,5})", re.IGNORECASE)
_AUDIO_RX = re.compile(r"Stream #\d+:\d+.*: Audio:", re.IGNORECASE)


def _readable_path(media: MediaInput) -> Optional[str]:
    """Return a usable filesystem path for an upload, or None."""
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


def _run(cmd: list[str]) -> Optional[subprocess.CompletedProcess]:
    """Run an external command with a timeout, swallowing all errors."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _probe_with_ffprobe(exe: str, path: str) -> Optional[ProbeResult]:
    """Run ffprobe and parse its JSON. Returns None if it produced nothing
    usable (caller may then try the ffmpeg fallback)."""
    proc = _run([
        exe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ])
    if proc is None or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None

    duration = 0.0
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    width = height = 0
    has_audio = has_video = False
    for stream in data.get("streams", []) or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            has_video = True
            if not width:
                try:
                    width = int(stream.get("width", 0) or 0)
                    height = int(stream.get("height", 0) or 0)
                except (TypeError, ValueError):
                    width = height = 0
            # Some containers expose duration only on the stream.
            if duration <= 0.0:
                try:
                    duration = float(stream.get("duration", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass
        elif codec_type == "audio":
            has_audio = True
            if duration <= 0.0:
                try:
                    duration = float(stream.get("duration", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass

    return ProbeResult(
        duration_s=max(0.0, duration),
        has_audio=has_audio,
        has_video=has_video,
        width=max(0, width),
        height=max(0, height),
        ok=True,
        error=None,
    )


def _probe_with_ffmpeg(exe: str, path: str) -> Optional[ProbeResult]:
    """Parse the ``Duration:`` / ``Stream`` lines from ffmpeg's stderr banner.

    ``ffmpeg -i <path>`` with no output spec exits non-zero ("At least one
    output file must be specified") but still prints the input analysis to
    stderr, which is all we need.
    """
    proc = _run([exe, "-hide_banner", "-i", path])
    if proc is None:
        return None
    text = (proc.stderr or "") + (proc.stdout or "")
    if not text:
        return None

    duration = 0.0
    m = _DURATION_RX.search(text)
    if m:
        try:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
            duration = hh * 3600 + mm * 60 + ss
        except (TypeError, ValueError):
            duration = 0.0

    width = height = 0
    has_video = False
    vm = _VIDEO_RX.search(text)
    if vm:
        has_video = True
        try:
            width = int(vm.group(1))
            height = int(vm.group(2))
        except (TypeError, ValueError):
            width = height = 0
    has_audio = _AUDIO_RX.search(text) is not None

    # If ffmpeg told us nothing at all about the streams, treat as a failure.
    if not (has_video or has_audio or duration > 0.0):
        return None

    return ProbeResult(
        duration_s=max(0.0, duration),
        has_audio=has_audio,
        has_video=has_video,
        width=max(0, width),
        height=max(0, height),
        ok=True,
        error=None,
    )


def probe(media: MediaInput) -> ProbeResult:
    """Probe an uploaded media file for duration / dimensions / stream presence.

    Returns ``ProbeResult(ok=False, error=...)`` for non-uploads, unreadable
    paths, or when no ffprobe/ffmpeg binary is available. Never raises.
    """
    path = _readable_path(media)
    if path is None:
        return ProbeResult(ok=False, error="no readable upload path")

    ffprobe = registry.ffprobe_exe()
    if ffprobe:
        result = _probe_with_ffprobe(ffprobe, path)
        if result is not None:
            return result
        # ffprobe present but unhelpful — fall through to ffmpeg below.

    ffmpeg = registry.ffmpeg_exe()
    if ffmpeg:
        result = _probe_with_ffmpeg(ffmpeg, path)
        if result is not None:
            return result
        return ProbeResult(ok=False, error="ffmpeg could not parse media")

    if not ffprobe:
        return ProbeResult(ok=False, error="no ffprobe/ffmpeg")
    return ProbeResult(ok=False, error="ffprobe could not parse media")
