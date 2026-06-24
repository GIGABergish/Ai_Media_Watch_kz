"""Lexical text-signal lane — the cheap, always-on first tier of the cascade.

Scans every textual source available on the bundle against the ScamDNA
dimension lexicons (and the risk-reducing negative markers), emitting a flat
list of :class:`AnalyzerHit` records. Each source is scanned separately so the
originating timestamp (audio segment / OCR frame) is preserved for the timeline.

Pure stdlib — no optional dependencies, never raises.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.config import clamp_score
from app.pipeline.contracts import AnalyzerHit, SignalBundle
from app.scoring.lexicons import (
    DIMENSION_PATTERNS,
    NEGATIVE_MARKERS,
    chip_for,
    find_matches,
)

# Sentinel dna_key emitted for risk-REDUCING matches; scoring/category treat
# "negative" specially (it subtracts rather than adds).
NEGATIVE_KEY = "negative"

_SENT_SPLIT = re.compile(r"[.!?\n]+")
_FRAGMENT_MAX = 160


def _surrounding_sentence(text: str, start: int) -> str:
    """Return the trimmed sentence/segment of ``text`` containing offset ``start``."""
    if not text:
        return ""
    lo = 0
    hi = len(text)
    for m in _SENT_SPLIT.finditer(text):
        if m.end() <= start:
            lo = m.end()
        elif m.start() >= start:
            hi = m.start()
            break
    fragment = text[lo:hi].strip()
    if len(fragment) > _FRAGMENT_MAX:
        fragment = fragment[:_FRAGMENT_MAX].rstrip() + "…"
    return fragment


def _scan_source(
    text: str,
    source: str,
    time_s: Optional[float],
    seen: set,
) -> List[AnalyzerHit]:
    """Scan one text blob across all dimensions, returning deduped hits."""
    hits: List[AnalyzerHit] = []
    if not text or not text.strip():
        return hits

    for key, phrases in DIMENSION_PATTERNS.items():
        for match in find_matches(text, phrases):
            chip = chip_for(match)
            dedup = (key, chip, source)
            if dedup in seen:
                continue
            seen.add(dedup)
            hits.append(AnalyzerHit(
                source=source,
                confidence=clamp_score(match.phrase.weight),
                signal=f"Обнаружено: {chip}",
                fragment=_surrounding_sentence(text, match.start),
                time_s=time_s,
                dna_keys=[key],
                tags=[chip],
            ))
    return hits


def analyze_text(bundle: SignalBundle) -> list[AnalyzerHit]:
    """Lexical scan of metadata, transcript segments and OCR hits.

    Three sources are scanned independently to keep timestamps:
      * ``Metadata`` — title + description + hashtags (``time_s=None``);
      * ``Audio``   — each transcript segment (``time_s=seg.start_s``);
      * ``OCR``     — each OCR hit (``time_s=hit.time_s``).

    Also scans :data:`NEGATIVE_MARKERS` over the combined text, emitting
    risk-reducing ``Metadata`` hits with ``dna_keys=["negative"]``.

    Appends ``"text"`` to ``bundle.lanes_run`` and returns the hit list.
    """
    bundle.lanes_run.append("text")

    seen: set[Tuple[str, str, str]] = set()
    hits: List[AnalyzerHit] = []

    # (a) Metadata -------------------------------------------------------- #
    m = bundle.media
    meta_parts = [m.title, m.description, " ".join(m.hashtags)]
    meta_text = "\n".join(p for p in meta_parts if p)
    hits.extend(_scan_source(meta_text, "Metadata", None, seen))

    # (b) Audio (per transcript segment) ---------------------------------- #
    for seg in bundle.transcript.segments:
        hits.extend(_scan_source(seg.text, "Audio", seg.start_s, seen))

    # (c) OCR (per frame) ------------------------------------------------- #
    for hit in bundle.ocr_hits:
        hits.extend(_scan_source(hit.text, "OCR", hit.time_s, seen))

    # Negative markers over the combined text (risk-reducing) ------------- #
    combined = bundle.all_text()
    neg_seen: set[str] = set()
    for match in find_matches(combined, NEGATIVE_MARKERS):
        chip = chip_for(match)
        if chip in neg_seen:
            continue
        neg_seen.add(chip)
        hits.append(AnalyzerHit(
            source="Metadata",
            confidence=clamp_score(match.phrase.weight),
            signal="Маркер образовательного/антимошеннического контекста",
            fragment=_surrounding_sentence(combined, match.start),
            time_s=None,
            dna_keys=[NEGATIVE_KEY],
            tags=[chip],
        ))

    return hits
