"""Weak-supervision labeller — turns the rule engine into a TEACHER.

The existing rule engine (``app.scoring``) is the source of distilled targets for
the learned model. This module runs the teacher chain over a
:class:`SignalBundle` and packages its verdict as a :class:`Label`:

    compute_scam_dna(bundle)
        -> compute_risk_score(bundle, dna)
        -> classify_category(bundle, dna, breakdown)

The label carries the overall ``risk`` (0..1), the 8 ScamDNA ``dimensions``
(0..1, ordered by :data:`DIMENSION_KEYS`), the coarse ``category`` machine key,
the binary ``is_scam`` flag and a sample ``weight`` derived from the teacher's
*margin* — its confidence expressed in the engine's own threshold space
(reusing :data:`app.config.settings.thresholds`, never re-deriving cut-offs).

``weak_label_from_text`` is the cold-input path: it assembles a minimal,
text-only bundle and runs the cheap always-on lanes (links -> text -> behavior)
so the teacher sees the same hits it would in the full pipeline, then labels it.

This is the dimension-oracle half of the distillation strategy (see DESIGN §1/§6):
synthetic ground truth owns ``risk``/``is_scam``/``category`` for generated rows,
while the teacher reliably supplies the soft ``dimensions`` on *clean* text. For
real, un-augmented inputs the teacher owns the whole label.
"""
from __future__ import annotations

from typing import List, Optional

from app.config import settings
from app.ml.types import DIMENSION_KEYS, Label
from app.pipeline.contracts import (
    AnalyzerHit,
    MediaInput,
    Platform,
    SignalBundle,
)
from app.scoring.category import classify_category
from app.scoring.risk_score import compute_risk_score
from app.scoring.scam_dna import compute_scam_dna

# Cheap, always-on first-tier lanes (pure stdlib, never raise). Order matters:
# links must run BEFORE behavior, because the behavior lane consumes the
# ``bundle.link_hits`` that the links lane fills.
from app.analyzers.behavior import analyze_behavior
from app.analyzers.links import extract_links
from app.analyzers.text_signals import analyze_text

# Smallest sample weight a teacher label may carry — a genuinely band-boundary
# (maximally ambiguous) example is down-weighted but never fully dropped, so the
# student still sees the hard cases.
_MIN_WEIGHT = 0.20


def _teacher_margin(score: float) -> float:
    """Teacher confidence in [_MIN_WEIGHT, 1.0] from the overall risk ``score``.

    Confidence = the teacher's distance to its nearest *decision* threshold,
    normalized so band-centre verdicts pull hard and band-boundary verdicts
    (genuinely ambiguous to the engine) are down-weighted.

    The decision thresholds are the engine's own internal cut-offs — medium /
    high / critical from ``settings.thresholds``. The outer bounds 0 and 100 are
    NOT decision points (a score pinned at 0 is the engine's *most* confident
    "benign" verdict, not an ambiguous one), so distance is measured only to the
    internal thresholds and then normalized by the local band's half-width. This
    yields ~1.0 deep inside the lowest/highest bands and at the 0/100 extremes,
    and ``_MIN_WEIGHT`` exactly on an internal threshold.

    Reuses ``settings.thresholds`` so "ambiguous to the engine" stays in lockstep
    with the rule engine's banding; no thresholds are duplicated here.
    """
    t = settings.thresholds
    # Internal decision thresholds + outer bounds (for band-width normalization).
    boundaries: List[float] = [0.0, float(t.medium), float(t.high),
                               float(t.critical), 100.0]
    internal = boundaries[1:-1]                    # medium, high, critical
    s = max(0.0, min(100.0, float(score)))

    # Locate the [lo, hi] band containing the score (for the local half-width).
    lo, hi = boundaries[0], boundaries[-1]
    for left, right in zip(boundaries, boundaries[1:]):
        if left <= s <= right:
            lo, hi = left, right
            break

    half = (hi - lo) / 2.0
    if half <= 0.0:
        return 1.0
    # Distance to the nearest INTERNAL threshold, normalized by the band's
    # half-width. Capped at 1.0 so deep-band / extreme scores saturate.
    dist = min(abs(s - b) for b in internal)
    return max(_MIN_WEIGHT, min(1.0, dist / half))


def weak_label(bundle: SignalBundle) -> Label:
    """Run the rule-engine teacher over ``bundle`` and package its verdict.

    Assumes the bundle already carries its analyzer ``hits`` / ``link_hits`` /
    ``visual_hits`` (the orchestrator or :func:`weak_label_from_text` fills
    them). Pure read of the teacher — never mutates the bundle.

    Returns a :class:`Label` with ``source="weak"``:
      * ``risk``       = ``breakdown.score / 100``;
      * ``dimensions`` = ``{key: value / 100}`` for all 8 :data:`DIMENSION_KEYS`;
      * ``category``   = the machine key from ``classify_category``;
      * ``is_scam``    = ``risk >= 0.5``;
      * ``weight``     = teacher margin (confidence in its own threshold space).
    """
    dna = compute_scam_dna(bundle)
    breakdown = compute_risk_score(bundle, dna)
    category, _category_ru = classify_category(bundle, dna, breakdown)

    by_key = {d.key: float(d.value) for d in dna}
    dimensions = {k: by_key.get(k, 0.0) / 100.0 for k in DIMENSION_KEYS}

    risk = float(breakdown.score) / 100.0

    return Label(
        risk=risk,
        dimensions=dimensions,
        category=category,
        is_scam=risk >= 0.5,
        source="weak",
        weight=_teacher_margin(breakdown.score),
    )


def weak_label_from_text(
    text: str,
    hashtags: Optional[List[str]] = None,
    platform: Optional[Platform] = None,
) -> Label:
    """Weak-label a raw text snippet (no media) via the cheap analyzer lanes.

    Builds a minimal text-only :class:`SignalBundle`
    (``MediaInput(source_type="text")`` with ``text`` placed in the description so
    every cheap lane sees it), runs the always-on first-tier lanes in dependency
    order — ``extract_links`` -> ``analyze_text`` -> ``analyze_behavior`` —
    appending their hits to ``bundle.hits`` so the teacher reads them, then
    delegates to :func:`weak_label`.

    This is the orchestrator-side / cold-input path used for real, un-augmented
    text (and by synth's clean-text dimension extraction).
    """
    media = MediaInput(
        source_type="text",
        description=text or "",
        hashtags=list(hashtags) if hashtags else [],
        platform=platform,
    )
    bundle = SignalBundle(media=media)

    # Run cheap lanes in order; each fills the bundle and returns its hits, which
    # we collect onto ``bundle.hits`` (the surface the teacher scores over).
    hits: List[AnalyzerHit] = []
    hits.extend(extract_links(bundle))      # fills bundle.link_hits
    hits.extend(analyze_text(bundle))       # lexical dimension/negative hits
    hits.extend(analyze_behavior(bundle))   # CTA + link-driven funnel hits
    for hit in hits:
        bundle.add_hit(hit)

    return weak_label(bundle)
