"""Behavioral-pattern lane — engagement / call-to-action tactics.

Detects the *funnel mechanics* of a scam rather than its claims: pushes into
messengers, referral hooks and urgency-driven CTAs. It consumes the link hits
already extracted by :mod:`app.analyzers.links` and additionally runs a small
set of CTA regexes over the bundle's combined text.

Feeds the ``urgency``, ``referral`` and ``messenger`` ScamDNA dimensions. Some
overlap with the text lane is intentional — ``scoring.scam_dna`` combines hits
with a saturating aggregate. Pure stdlib, never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import clamp_score
from app.pipeline.contracts import AnalyzerHit, SignalBundle

_FRAGMENT_MAX = 160


@dataclass(frozen=True)
class _CtaRule:
    pattern: str
    weight: int
    signal: str          # RU description
    dna_keys: Tuple[str, ...]
    tag: str             # RU finding chip


# Call-to-action / engagement regexes (case-insensitive, unicode).
_CTA_RULES: List[_CtaRule] = [
    _CtaRule(r"пиши(?:те)?\s*[\"«]?\+", 82,
             "Призыв написать в личные сообщения по номеру",
             ("messenger",), "«пиши +»"),
    _CtaRule(r"пиши(?:те)?\s+в\s+директ", 80,
             "Перевод аудитории в директ", ("messenger",), "«пиши в директ»"),
    _CtaRule(r"пиши(?:те)?\s+в\s+(?:лс|личк)", 78,
             "Перевод аудитории в личные сообщения",
             ("messenger",), "«пиши в личку»"),
    _CtaRule(r"осталось\s+\d+\s+мест", 85,
             "Искусственный дефицит мест", ("urgency",), "«осталось N мест»"),
    _CtaRule(r"(?:успей|успейте)\s+до", 80,
             "Дедлайн для давления срочности", ("urgency",), "«успей до»"),
    _CtaRule(r"только\s+сегодня", 78,
             "Ограничение по времени", ("urgency",), "«только сегодня»"),
    _CtaRule(r"последн(?:ий|ие)\s+(?:шанс|мест)", 80,
             "Призыв действовать немедленно", ("urgency",), "«последний шанс»"),
    _CtaRule(r"привед(?:и|ите)\s+\d+", 88,
             "Реферальное условие — привести участников",
             ("referral",), "«приведи N человек»"),
    _CtaRule(r"пригласи(?:те)?\s+(?:друг|знаком)", 80,
             "Призыв приглашать новых участников",
             ("referral",), "«пригласи друзей»"),
    _CtaRule(r"по\s+(?:моей|этой)\s+ссылке", 76,
             "Регистрация по реферальной ссылке",
             ("referral",), "«по моей ссылке»"),
    _CtaRule(r"жми\s+(?:на\s+)?ссылк", 72,
             "Призыв перейти по ссылке", ("messenger",), "«жми на ссылку»"),
]

# Link kind -> (ScamDNA dims, confidence, RU signal prefix).
_LINK_BEHAVIOR = {
    "telegram": (("messenger",), 80, "Воронка в Telegram"),
    "whatsapp": (("messenger",), 80, "Воронка в WhatsApp"),
    "phone": (("messenger",), 60, "Призыв связаться по телефону"),
    "promocode": (("referral",), 82, "Использование промокода / реферального кода"),
}


def _trim(text: str) -> str:
    text = text.strip()
    if len(text) > _FRAGMENT_MAX:
        text = text[:_FRAGMENT_MAX].rstrip() + "…"
    return text


def _locate_time(bundle: SignalBundle, matched: str) -> Optional[float]:
    """Best-effort timestamp: earliest transcript segment / OCR hit containing
    ``matched`` (case-insensitive). Returns ``None`` when not locatable."""
    needle = matched.strip().lower()
    if not needle:
        return None
    best: Optional[float] = None
    for seg in bundle.transcript.segments:
        if needle in (seg.text or "").lower():
            if best is None or seg.start_s < best:
                best = seg.start_s
    for hit in bundle.ocr_hits:
        if needle in (hit.text or "").lower():
            if best is None or hit.time_s < best:
                best = hit.time_s
    return best


def analyze_behavior(bundle: SignalBundle) -> list[AnalyzerHit]:
    """Emit ``Behavior``-source hits for CTA / engagement tactics.

    Combines (1) link hits already on the bundle (messenger / referral funnels)
    with (2) CTA regexes over :meth:`SignalBundle.all_text`. Appends
    ``"behavior"`` to ``bundle.lanes_run`` and returns the hit list.
    """
    bundle.lanes_run.append("behavior")

    hits: List[AnalyzerHit] = []
    seen: set[Tuple[str, str]] = set()   # (tag, source-discriminator)

    # (1) Link-hit driven funnels ---------------------------------------- #
    for link in bundle.link_hits:
        meta = _LINK_BEHAVIOR.get(link.kind)
        if not meta:
            continue
        dna_keys, conf, prefix = meta
        dedup = (link.kind, link.value.lower())
        if dedup in seen:
            continue
        seen.add(dedup)
        hits.append(AnalyzerHit(
            source="Behavior",
            confidence=clamp_score(conf),
            signal=f"{prefix}: {link.value}",
            fragment=_trim(link.value),
            time_s=None,
            dna_keys=list(dna_keys),
            tags=[link.value],
        ))

    # (2) CTA regexes over combined text --------------------------------- #
    text = bundle.all_text()
    if text:
        for rule in _CTA_RULES:
            for m in re.finditer(rule.pattern, text, re.IGNORECASE | re.UNICODE):
                matched = m.group(0)
                dedup = (rule.tag, matched.strip().lower())
                if dedup in seen:
                    continue
                seen.add(dedup)
                hits.append(AnalyzerHit(
                    source="Behavior",
                    confidence=clamp_score(rule.weight),
                    signal=rule.signal,
                    fragment=_trim(matched),
                    time_s=_locate_time(bundle, matched),
                    dna_keys=list(rule.dna_keys),
                    tags=[rule.tag],
                ))

    return hits
