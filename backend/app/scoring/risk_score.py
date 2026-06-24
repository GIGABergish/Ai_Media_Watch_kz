"""Risk Score scorer — blends the ScamDNA dimensions into the five weighted
components and the single overall 0..100 risk score.

The five components mirror the "Формула Risk Score" block in the frontend
settings page and are weighted by ``settings.weights``. Educational / anti-fraud
content (negative markers) damps the overall score so that explainer videos and
fraud-awareness clips land in the low band.
"""
from __future__ import annotations

from typing import Dict, List

from app.api.schemas import ScamDNADimension
from app.config import clamp_score, risk_level, settings
from app.pipeline.contracts import RiskBreakdown, SignalBundle
from app.scoring.lexicons import DIMENSION_BY_KEY
from app.scoring.scam_dna import dna_map


def _sat(*values: float) -> float:
    """Saturating blend of 0..100 contributions (diminishing returns).

    Sorted descending so the strongest term anchors the result; each further
    term adds only its share of the remaining headroom. Returns 0..100.
    """
    acc = 0.0
    for x in sorted((max(0.0, v) for v in values), reverse=True):
        acc = acc + x * (1.0 - acc / 100.0)
    return acc


# Human-readable RU reasons for the dominant dimension driving the score.
_REASON_BY_KEY = {
    "profit": "Обещания гарантированной прибыли без предупреждений о рисках",
    "urgency": "Агрессивное давление срочности и искусственный дефицит",
    "gambling": "Явные маркеры онлайн-казино и азартных игр",
    "referral": "Признаки реферальной схемы и приглашения новых участников",
    "messenger": "Воронка увода аудитории в мессенджер в обход модерации",
    "visual": "Визуальные манипуляции: поддельные скриншоты выплат и графики",
    "reused": "Совпадение нарратива с известными мошенническими шаблонами",
    "hashtags": "Кластеры хэштегов, нацеленных на ищущих лёгкий заработок",
}


def _negative_total(bundle: SignalBundle) -> float:
    """Saturating total of confidences of negative (risk-reducing) hits."""
    vals = [
        float(h.confidence)
        for h in bundle.hits
        if "negative" in h.dna_keys
    ]
    return _sat(*vals) if vals else 0.0


def _main_reason(
    bundle: SignalBundle,
    dm: Dict[str, int],
    overall: float,
    neg: float,
) -> str:
    """Compose a RU sentence describing the strongest risk driver."""
    if overall < settings.thresholds.medium:
        if neg > 0:
            return ("Контент похож на образовательный / антимошеннический: "
                    "присутствуют предупреждения о рисках")
        return "Явных признаков мошеннической схемы не обнаружено"

    # Pick the dimension with the highest value as the headline driver.
    best_key = max(dm, key=lambda k: dm[k]) if dm else ""
    base = _REASON_BY_KEY.get(best_key) or (
        DIMENSION_BY_KEY[best_key].default_desc if best_key in DIMENSION_BY_KEY
        else "Совокупность подозрительных признаков"
    )

    # Prefer a concrete fragment from the strongest hit feeding that dimension.
    best_hit = None
    for h in bundle.hits:
        if best_key in h.dna_keys:
            if best_hit is None or h.confidence > best_hit.confidence:
                best_hit = h
    if best_hit is not None:
        frag = (best_hit.fragment or "").strip()
        if frag:
            return f"{base}: «{frag}»"
        if best_hit.signal:
            return best_hit.signal
    return base


def compute_risk_score(
    bundle: SignalBundle,
    dna: List[ScamDNADimension],
) -> RiskBreakdown:
    """Combine ScamDNA dimensions into the weighted overall risk breakdown."""
    dm = dna_map(dna)

    def g(key: str) -> float:
        return float(dm.get(key, 0))

    # Five weighted components (each 0..100).
    text_speech = _sat(g("profit"), 0.85 * g("urgency"))
    visual = _sat(g("gambling"), g("visual"))
    metadata_links = _sat(0.85 * g("hashtags"), 0.8 * g("messenger"))
    behavior = _sat(g("referral"), 0.7 * g("urgency"), 0.6 * g("messenger"))
    db_similarity = g("reused")

    w = settings.weights
    overall = (
        text_speech * w.text_speech
        + visual * w.visual
        + metadata_links * w.metadata_links
        + behavior * w.behavior
        + db_similarity * w.db_similarity
    )

    # Negative damping — educational / anti-fraud content scores low.
    neg = _negative_total(bundle)
    overall *= (1.0 - min(0.55, neg / 180.0))

    main_reason = _main_reason(bundle, dm, overall, neg)

    return RiskBreakdown(
        score=clamp_score(overall),
        level=risk_level(overall),
        text_speech=clamp_score(text_speech),
        visual=clamp_score(visual),
        metadata_links=clamp_score(metadata_links),
        behavior=clamp_score(behavior),
        db_similarity=clamp_score(db_similarity),
        main_reason=main_reason,
    )
