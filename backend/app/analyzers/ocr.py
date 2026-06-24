"""OCR lane — reads on-screen text from sampled keyframes via Tesseract.

Recognised text (promo codes, casino names, messenger handles, fake bank
overlays) feeds the cheap lexical analyzers and the timeline. The whole lane is
optional: without Tesseract / Pillow / keyframes it degrades to a no-op and
flags ``bundle.degradation.ocr`` so downstream scoring stays valid.

Heavy deps (``pytesseract``, ``PIL``) are imported lazily inside the function
and wrapped so a missing dependency never raises out of this module.
"""
from __future__ import annotations

from typing import List

from app.config import settings
from app.models import registry
from app.pipeline.contracts import Keyframe, OcrHit, SignalBundle

# Minimum length of an aggregated frame string worth keeping. Tesseract noise on
# benign footage is usually one or two stray glyphs; require something textual.
_MIN_TEXT_LEN = 3


def run_ocr(bundle: SignalBundle, keyframes: List[Keyframe]) -> None:
    """Fill ``bundle.ocr_hits`` with text recognised on each keyframe.

    Degrades to a no-op (and sets ``bundle.degradation.ocr``) when OCR is
    disabled, Tesseract is unavailable, or there are no keyframes to read.
    Never raises.
    """
    if not settings.enable_ocr or not registry.tesseract_available() or not keyframes:
        bundle.degradation.ocr = True
        if not settings.enable_ocr:
            bundle.degradation.notes.append("OCR отключён в настройках.")
        elif not registry.tesseract_available():
            bundle.degradation.notes.append(
                "OCR недоступен: Tesseract не установлен."
            )
        else:
            bundle.degradation.notes.append("OCR пропущен: нет кадров для анализа.")
        return

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001 - degrade, never propagate
        bundle.degradation.ocr = True
        bundle.degradation.notes.append(f"OCR недоступен: {exc!r}")
        return

    try:
        for kf in keyframes:
            image = _frame_image(kf, Image)
            if image is None:
                continue
            hit = _read_frame(kf, image, pytesseract)
            if hit is not None:
                bundle.ocr_hits.append(hit)
        bundle.lanes_run.append("ocr")
    except Exception as exc:  # noqa: BLE001 - any failure degrades the lane
        bundle.degradation.ocr = True
        bundle.degradation.notes.append(f"OCR прерван: {exc!r}")


def _frame_image(kf: Keyframe, Image) -> object:
    """Return a PIL image for the keyframe (in-memory or opened from disk)."""
    if kf.image is not None:
        return kf.image
    if kf.path:
        try:
            return Image.open(kf.path)
        except Exception:  # noqa: BLE001 - skip unreadable frame
            return None
    return None


def _read_frame(kf: Keyframe, image: object, pytesseract) -> OcrHit | None:
    """Run Tesseract on one frame and build an OcrHit, or None if trivial."""
    try:
        data = pytesseract.image_to_data(
            image,
            lang=settings.tesseract_lang,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:  # noqa: BLE001 - skip frames Tesseract chokes on
        return None

    words: List[str] = []
    confs: List[float] = []
    raw_words = data.get("text", [])
    raw_confs = data.get("conf", [])
    for word, conf in zip(raw_words, raw_confs):
        token = (word or "").strip()
        if not token:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c <= 0:
            continue
        words.append(token)
        confs.append(c)

    text = " ".join(words).strip()
    if len(text) < _MIN_TEXT_LEN or not confs:
        return None

    confidence = max(0.0, min(100.0, sum(confs) / len(confs)))
    return OcrHit(time_s=kf.time_s, text=text, confidence=confidence)
