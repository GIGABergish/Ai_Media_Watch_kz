"""ASR lane — speech-to-text for the audio track.

Fills ``bundle.transcript`` with per-segment timestamps so the Audio-source
timeline events land at the correct times. This module does NOT create
``AnalyzerHit``s — ``analyzers.text_signals`` mines the transcript text.

Defensive contract: every optional dependency (``faster_whisper`` / ``whisper``,
``torch``) is imported lazily inside the function. A missing dependency or any
runtime error degrades gracefully — ``bundle.degradation.asr`` is set, a short
Russian note is recorded, and the transcript stays ``source="none"``. It never
raises out of :func:`run_asr`.
"""
from __future__ import annotations

from typing import List, Optional

from app.config import settings
from app.models import registry
from app.pipeline.contracts import (
    AudioTrack,
    SignalBundle,
    Transcript,
    TranscriptSegment,
)


def run_asr(bundle: SignalBundle, audio: Optional[AudioTrack]) -> None:
    """Transcribe the audio track into ``bundle.transcript``.

    1. A pre-supplied transcript (platform captions) short-circuits ASR.
    2. If ASR is disabled, unavailable, or there is no audio -> degrade.
    3. Otherwise load Whisper once (cached) and transcribe with timestamps.
    """
    # --- (1) Provided transcript wins — no model needed. ------------------ #
    provided = bundle.media.provided_transcript
    if provided and provided.strip():
        text = provided.strip()
        bundle.transcript = Transcript(
            segments=[TranscriptSegment(start_s=0.0, end_s=0.0, text=text)],
            full_text=text,
            language="",
            source="provided",
        )
        bundle.lanes_run.append("asr")
        return

    # --- (2) Cannot or should not run ASR — degrade gracefully. ----------- #
    if not settings.enable_asr:
        bundle.degradation.asr = True
        bundle.degradation.notes.append(
            "Распознавание речи отключено в настройках — звук не анализировался."
        )
        return
    if not registry.capabilities()["asr"]:
        bundle.degradation.asr = True
        bundle.degradation.notes.append(
            "Модель распознавания речи недоступна — звуковая дорожка пропущена."
        )
        return
    if audio is None or not getattr(audio, "path", None):
        bundle.degradation.asr = True
        bundle.degradation.notes.append(
            "Звуковая дорожка не извлечена — речь не распознавалась."
        )
        return

    # --- (3) Real transcription. ------------------------------------------ #
    try:
        segments, full_text, language = _transcribe(audio.path)
        bundle.transcript = Transcript(
            segments=segments,
            full_text=full_text,
            language=language,
            source="whisper",
        )
        bundle.lanes_run.append("asr")
    except Exception as exc:  # noqa: BLE001 - degrade, never propagate
        bundle.degradation.asr = True
        bundle.degradation.notes.append(
            f"Ошибка распознавания речи — звук пропущен ({type(exc).__name__})."
        )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _transcribe(path: str):
    """Run the cached Whisper backend over ``path``.

    Prefers ``faster_whisper`` (lighter, faster) and falls back to the
    reference ``whisper`` package. Returns ``(segments, full_text, language)``.
    Raises on any failure — :func:`run_asr` catches and degrades.
    """
    backend = registry.cached("whisper", _build_model)
    if backend is None:
        err = registry.load_error("whisper") or "модель не загрузилась"
        raise RuntimeError(f"Whisper unavailable: {err}")

    kind, model = backend
    if kind == "faster_whisper":
        return _transcribe_faster(model, path)
    return _transcribe_openai(model, path)


def _build_model():
    """Load a Whisper model once. Tagged with its backend kind.

    Tried inside ``registry.cached`` so a load failure is cached as a miss and
    surfaced via ``registry.load_error('whisper')`` rather than retried.
    """
    model_name = settings.whisper_model
    device = settings.whisper_device

    if registry.has_module("faster_whisper"):
        from faster_whisper import WhisperModel  # type: ignore

        # int8 keeps CPU inference cheap; float16 is unsafe on CPU.
        compute_type = "float16" if device != "cpu" else "int8"
        model = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        return ("faster_whisper", model)

    if registry.has_module("whisper"):
        import whisper  # type: ignore

        model = whisper.load_model(model_name, device=device)
        return ("whisper", model)

    raise RuntimeError("no whisper backend installed")


def _transcribe_faster(model, path: str):
    """faster-whisper path — segments stream with native start/end seconds."""
    segments_iter, info = model.transcribe(path)
    segments: List[TranscriptSegment] = []
    parts: List[str] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_s=float(seg.start or 0.0),
                end_s=float(seg.end or 0.0),
                text=text,
            )
        )
        parts.append(text)
    language = getattr(info, "language", "") or ""
    return segments, " ".join(parts), language


def _transcribe_openai(model, path: str):
    """openai-whisper path — result dict carries per-segment timestamps."""
    result = model.transcribe(path)
    segments: List[TranscriptSegment] = []
    parts: List[str] = []
    for seg in result.get("segments", []) or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_s=float(seg.get("start", 0.0) or 0.0),
                end_s=float(seg.get("end", 0.0) or 0.0),
                text=text,
            )
        )
        parts.append(text)

    full_text = (result.get("text") or " ".join(parts)).strip()
    language = result.get("language", "") or ""
    return segments, full_text, language
