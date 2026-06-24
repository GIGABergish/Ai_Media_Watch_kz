"""Timeline scorer — turns time-stamped signals into the ordered list of
``TimelineEvent`` the UI renders under the "Таймлайн сигналов" panel.

Sources of events:
  * ``bundle.hits`` whose ``time_s`` is not None (text/audio/OCR/behavior hits
    that an analyzer was able to anchor to a moment in the video);
  * ``bundle.visual_hits`` (CLIP zero-shot detections) — each becomes a
    ``Visual`` source event with the Russian label and confidence ``score*100``.

Events are sorted by time, near-duplicate signals from the same source within a
1-second window are collapsed, the most severe ~8 are kept, and each is rendered
as a ``TimelineEvent`` with an ``MM:SS`` label.
"""
from __future__ import annotations

from typing import List, Optional

from app.api.schemas import TimelineEvent
from app.config import clamp_score, severity_from_confidence
from app.pipeline.contracts import SignalBundle

# How many events the UI comfortably shows.
MAX_EVENTS = 8
# Two events from the same source within this many seconds are "the same".
DEDUP_WINDOW_S = 1.0


def mmss(seconds: float) -> str:
    """Format a number of seconds as ``"MM:SS"`` (minutes are not zero-capped)."""
    total = int(round(max(0.0, seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _severity_rank(confidence: float) -> int:
    """Order severities so the *strongest* events survive de-duplication / trim."""
    order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    return order.get(severity_from_confidence(confidence), 0)


class _Candidate:
    """A normalized timeline candidate before it becomes a ``TimelineEvent``."""

    __slots__ = ("time_s", "source", "signal", "confidence")

    def __init__(self, time_s: float, source: str, signal: str, confidence: float):
        self.time_s = float(time_s)
        self.source = source
        self.signal = signal
        self.confidence = float(confidence)


def _collect(bundle: SignalBundle) -> List[_Candidate]:
    """Gather all time-anchored signals from the bundle as candidates."""
    out: List[_Candidate] = []

    # Time-anchored analyzer hits (audio / OCR / behavior / text with a moment).
    for h in bundle.hits:
        if h.time_s is None:
            continue
        signal = (h.signal or h.fragment or "").strip()
        if not signal:
            continue
        out.append(_Candidate(h.time_s, h.source, signal, h.confidence))

    # CLIP visual detections.
    for v in bundle.visual_hits:
        label = (v.label_ru or v.label or "").strip()
        if not label:
            continue
        out.append(_Candidate(v.time_s, "Visual", label, v.score * 100.0))

    return out


def _dedup(candidates: List[_Candidate]) -> List[_Candidate]:
    """Collapse near-identical events: same source within ``DEDUP_WINDOW_S``.

    Among collisions the most severe (then highest-confidence) candidate wins.
    Input is assumed sorted by ``time_s`` ascending.
    """
    kept: List[_Candidate] = []
    for cand in candidates:
        merged = False
        for i, existing in enumerate(kept):
            same_source = existing.source == cand.source
            close = abs(existing.time_s - cand.time_s) <= DEDUP_WINDOW_S
            same_signal = existing.signal.lower() == cand.signal.lower()
            if same_source and (close or same_signal):
                # Keep whichever is stronger.
                better = (
                    _severity_rank(cand.confidence) > _severity_rank(existing.confidence)
                    or (
                        _severity_rank(cand.confidence) == _severity_rank(existing.confidence)
                        and cand.confidence > existing.confidence
                    )
                )
                if better:
                    kept[i] = cand
                merged = True
                break
        if not merged:
            kept.append(cand)
    return kept


def build_timeline(bundle: SignalBundle) -> List[TimelineEvent]:
    """Build the ordered, de-duplicated, severity-trimmed timeline of events."""
    candidates = _collect(bundle)
    if not candidates:
        return []

    candidates.sort(key=lambda c: c.time_s)
    candidates = _dedup(candidates)

    # Keep the most severe ~MAX_EVENTS, then restore chronological order.
    if len(candidates) > MAX_EVENTS:
        candidates.sort(
            key=lambda c: (_severity_rank(c.confidence), c.confidence),
            reverse=True,
        )
        candidates = candidates[:MAX_EVENTS]
        candidates.sort(key=lambda c: c.time_s)

    events: List[TimelineEvent] = []
    for i, c in enumerate(candidates):
        conf = clamp_score(c.confidence)
        events.append(
            TimelineEvent(
                id=f"t{i + 1}",
                time=mmss(c.time_s),
                timeSeconds=int(round(c.time_s)),
                source=c.source,  # SignalSource literal, set by the analyzers
                signal=c.signal,
                confidence=conf,
                severity=severity_from_confidence(conf),  # type: ignore[arg-type]
            )
        )
    return events


__all__ = ["build_timeline", "mmss"]
