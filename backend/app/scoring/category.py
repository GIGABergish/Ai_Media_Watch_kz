"""Category classifier — maps the analyzed signals to a single violation
category (machine key + Russian label) using the ordered ``CATEGORY_RULES``.

Educational / anti-fraud content is evaluated first when negative markers carry
significant weight, so explainer clips are never mislabelled as scams. Otherwise
rules fire in declared order: the first whose DNA thresholds and (optional)
keyword conditions all hold wins. With no match, the score band decides between
the low-risk and the generic suspicious fallback.
"""
from __future__ import annotations

from typing import List, Tuple

from app.api.schemas import ScamDNADimension
from app.config import settings
from app.pipeline.contracts import RiskBreakdown, SignalBundle
from app.scoring.lexicons import (
    CATEGORY_RULES,
    DEFAULT_CATEGORY,
    LOW_RISK_CATEGORY,
    CategoryRule,
)
from app.scoring.scam_dna import dna_map

# A negative weight at/above this saturated total is "significant" enough to
# route the case through the educational rules first.
_SIGNIFICANT_NEGATIVE = 35.0


def _negative_total(bundle: SignalBundle) -> float:
    """Saturating total of confidences of negative (risk-reducing) hits."""
    acc = 0.0
    vals = sorted(
        (float(h.confidence) for h in bundle.hits if "negative" in h.dna_keys),
        reverse=True,
    )
    for x in vals:
        acc = acc + x * (1.0 - acc / 100.0)
    return acc


def _rule_matches(
    rule: CategoryRule,
    dm: dict,
    text_lc: str,
) -> bool:
    """A rule matches when all DNA thresholds hold AND keyword check passes."""
    for key, threshold in rule.min_dna:
        if dm.get(key, 0) < threshold:
            return False
    if rule.any_keywords:
        if not any(kw in text_lc for kw in rule.any_keywords):
            return False
    return True


def classify_category(
    bundle: SignalBundle,
    dna: List[ScamDNADimension],
    breakdown: RiskBreakdown,
) -> Tuple[str, str]:
    """Return ``(category, categoryRu)`` for the analyzed media."""
    dm = dna_map(dna)
    text_lc = bundle.all_text().lower()

    has_negative = any("negative" in h.dna_keys for h in bundle.hits)
    significant_negative = (
        has_negative and _negative_total(bundle) >= _SIGNIFICANT_NEGATIVE
    )

    # When educational / anti-fraud intent dominates, try those rules first.
    if significant_negative:
        for rule in CATEGORY_RULES:
            if rule.is_educational and _rule_matches(rule, dm, text_lc):
                return (rule.category, rule.categoryRu)

    # Ordered evaluation — first match wins.
    for rule in CATEGORY_RULES:
        if _rule_matches(rule, dm, text_lc):
            return (rule.category, rule.categoryRu)

    if breakdown.score < settings.thresholds.medium:
        return LOW_RISK_CATEGORY
    return DEFAULT_CATEGORY
