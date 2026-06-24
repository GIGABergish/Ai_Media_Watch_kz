"""Lazy model & capability registry.

Heavy ML libraries (torch/whisper/open-clip/pytesseract) and external binaries
(ffmpeg) are optional. This module is the ONE place that:

  * discovers whether each optional capability is available (cheap, import-free
    probing via ``importlib.util.find_spec``),
  * locates an ffmpeg binary (prefers the pip-installed ``imageio-ffmpeg`` wheel
    so the engine works with no system ffmpeg),
  * caches expensive singletons (loaded models) so they are built at most once.

Analyzers call ``cached("whisper", factory)`` etc.; they never import torch at
module top-level. Nothing here raises on missing deps — callers degrade.
"""
from __future__ import annotations

import importlib.util
import shutil
from functools import lru_cache
from typing import Callable, Dict, Optional

from app.config import settings

_CACHE: Dict[str, object] = {}


def cached(key: str, factory: Callable[[], object]) -> object:
    """Build ``factory()`` once and reuse it. Returns None and caches the miss
    if the factory raises (so we don't retry a broken load every request)."""
    if key in _CACHE:
        return _CACHE[key]
    try:
        _CACHE[key] = factory()
    except Exception as exc:  # noqa: BLE001 - degrade, never propagate
        _CACHE[key] = None
        _CACHE[f"{key}__error"] = repr(exc)
    return _CACHE[key]


def load_error(key: str) -> Optional[str]:
    return _CACHE.get(f"{key}__error")  # type: ignore[return-value]


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg executable, or None. Prefers imageio-ffmpeg's bundled
    binary so no system install is required."""
    if has_module("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg  # type: ignore
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe_exe() -> Optional[str]:
    """Path to ffprobe if present (not bundled by imageio-ffmpeg)."""
    return shutil.which("ffprobe")


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    if not has_module("pytesseract"):
        return False
    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - binary missing
        return False


def capabilities() -> Dict[str, bool]:
    """Snapshot of which lanes can actually run right now (flags ∧ deps)."""
    ff = ffmpeg_exe() is not None
    # ASR can run from a bundled audio path; needs whisper(+torch) and ffmpeg.
    asr = settings.enable_asr and ff and (
        has_module("whisper") or has_module("faster_whisper")
    )
    ocr = settings.enable_ocr and tesseract_available() and has_module("PIL")
    vision = settings.enable_vision and (
        has_module("open_clip") or has_module("clip")
    ) and has_module("torch") and has_module("PIL")
    return {
        "ffmpeg": ff,
        "asr": asr,
        "ocr": ocr,
        "vision": vision,
        "pillow": has_module("PIL"),
    }


def engine_mode() -> str:
    """'full' = all three ML lanes available, 'hybrid' = some, 'lite' = none."""
    caps = capabilities()
    ml = [caps["asr"], caps["ocr"], caps["vision"]]
    if all(ml):
        return "full"
    if any(ml):
        return "hybrid"
    return "lite"
