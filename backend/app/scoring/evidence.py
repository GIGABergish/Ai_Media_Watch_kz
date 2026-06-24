"""Evidence scorer — assembles the up-to-six ``EvidenceCard`` blocks the UI
renders under "Доказательная база".

One card per :data:`EvidenceType`, emitted ONLY when there is supporting
evidence in the bundle:

  ``audio``      Аудио / Речь          — Audio-source hits / transcript snippet
  ``ocr``        OCR / Текст на экране  — OCR-source hits / ``bundle.ocr_hits``
  ``visual``     Визуальные признаки    — ``bundle.visual_hits``
  ``metadata``   Описание и хэштеги     — Metadata-source hits (hashtags/desc)
  ``links``      Внешние ссылки         — ``bundle.link_hits``
  ``engagement`` Признаки вовлечения    — Behavior-source hits

Each card carries the strongest contributing confidence, a short Russian quote
(the matched fragment in « »), a tailored 1-2 sentence Russian explanation, the
timestamp of the strongest time-anchored hit, and up to six finding chips.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.api.schemas import EvidenceCard, ScamDNADimension
from app.config import clamp_score
from app.pipeline.contracts import AnalyzerHit, SignalBundle
from app.scoring.lexicons import DIMENSION_BY_KEY

MAX_FINDINGS = 6

# Human titles per evidence type (RU).
TITLES: Dict[str, str] = {
    "audio": "Аудио / Речь",
    "ocr": "OCR / Текст на экране",
    "visual": "Визуальные признаки",
    "metadata": "Описание и хэштеги",
    "links": "Внешние ссылки",
    "engagement": "Признаки вовлечения",
}

# Per-type fallback explanation when no dimension-specific text applies.
DEFAULT_EXPLANATIONS: Dict[str, str] = {
    "audio": "В звуковой дорожке распознаны фразы, характерные для мошеннических "
             "схем заработка.",
    "ocr": "На экране распознан текст с признаками мошеннического предложения.",
    "visual": "Кадры содержат визуальные маркеры, типичные для схем казино и "
              "фейковых доказательств дохода.",
    "metadata": "В описании и хэштегах присутствуют формулировки, нацеленные на "
                "аудиторию, ищущую лёгкий заработок.",
    "links": "Контент уводит аудиторию на внешние ресурсы и в мессенджеры в "
             "обход модерации площадки.",
    "engagement": "Зафиксированы приёмы вовлечения и давления на пользователя, "
                  "подталкивающие к быстрому действию.",
}

# Dimension-tailored explanations — keyed by ScamDNA dimension. Picked from the
# strongest contributing hit's ``dna_keys`` so each card explains the concrete
# risk it represents.
DNA_EXPLANATIONS: Dict[str, str] = {
    "profit": "Звучат обещания гарантированного дохода без предупреждений о "
              "рисках — ключевой признак инвестиционного развода.",
    "urgency": "Используется искусственная срочность и дефицит, чтобы не дать "
               "жертве времени на проверку предложения.",
    "gambling": "Присутствуют маркеры азартных игр и онлайн-казино — "
                "признак продвижения нелегального игорного бизнеса.",
    "referral": "Прослеживается реферальная/партнёрская механика с промокодами "
                "и приглашением новых участников — типично для финансовых пирамид.",
    "messenger": "Аудиторию уводят в закрытые каналы Telegram/WhatsApp в обход "
                 "модерации площадки, где продолжается обработка жертвы.",
    "visual": "Демонстрируются поддельные скриншоты выплат и нереалистичные "
              "графики доходности как ложное доказательство заработка.",
    "reused": "Нарратив совпадает с известными шаблонами мошеннических роликов "
              "из базы знаний.",
    "hashtags": "Подобраны хэштеги, таргетирующие людей в поиске лёгкого "
                "заработка, для расширения охвата схемы.",
}


def mmss(seconds: float) -> str:
    """Format a number of seconds as ``"MM:SS"``."""
    total = int(round(max(0.0, seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _quote(text: str) -> str:
    """Wrap a short fragment in Russian guillemets, trimming overly long quotes."""
    snippet = " ".join((text or "").split())
    if len(snippet) > 160:
        snippet = snippet[:157].rstrip() + "…"
    return f"«{snippet}»" if snippet else ""


def _explanation(card_type: str, hits: List[AnalyzerHit]) -> str:
    """Pick a dimension-tailored RU explanation from the strongest hit, else the
    per-type default."""
    strongest = max(hits, key=lambda h: h.confidence, default=None)
    if strongest:
        for key in strongest.dna_keys:
            if key in DNA_EXPLANATIONS:
                return DNA_EXPLANATIONS[key]
    return DEFAULT_EXPLANATIONS.get(card_type, "")


def _findings(hits: List[AnalyzerHit], extra: List[str]) -> List[str]:
    """Collect up to ``MAX_FINDINGS`` unique chips from hit tags + extras."""
    out: List[str] = []
    seen = set()
    for chip in extra:
        norm = chip.strip()
        low = norm.lower()
        if norm and low not in seen:
            seen.add(low)
            out.append(norm)
    for h in hits:
        for tag in h.tags:
            norm = (tag or "").strip()
            low = norm.lower()
            if norm and low not in seen:
                seen.add(low)
                out.append(norm)
                if len(out) >= MAX_FINDINGS:
                    return out
    return out[:MAX_FINDINGS]


def _strongest_timestamp(hits: List[AnalyzerHit]) -> Optional[str]:
    """``MM:SS`` of the highest-confidence time-anchored hit, if any."""
    timed = [h for h in hits if h.time_s is not None]
    if not timed:
        return None
    best = max(timed, key=lambda h: h.confidence)
    return mmss(best.time_s)


def _card(
    card_type: str,
    hits: List[AnalyzerHit],
    *,
    fragment: str,
    confidence: float,
    extra_findings: Optional[List[str]] = None,
) -> EvidenceCard:
    """Assemble a single ``EvidenceCard`` from its contributing hits."""
    return EvidenceCard(
        type=card_type,  # type: ignore[arg-type]
        title=TITLES[card_type],
        confidence=clamp_score(confidence),
        fragment=fragment,
        explanation=_explanation(card_type, hits),
        timestamp=_strongest_timestamp(hits),
        findings=_findings(hits, extra_findings or []),
    )


def _hits_by_source(bundle: SignalBundle, source: str) -> List[AnalyzerHit]:
    return [h for h in bundle.hits if h.source == source]


def _max_conf(hits: List[AnalyzerHit]) -> float:
    return max((h.confidence for h in hits), default=0.0)


def _best_fragment(hits: List[AnalyzerHit]) -> str:
    """Quote from the strongest hit's fragment (or signal)."""
    if not hits:
        return ""
    best = max(hits, key=lambda h: h.confidence)
    return _quote(best.fragment or best.signal)


def build_evidence(
    bundle: SignalBundle, dna: List[ScamDNADimension]
) -> List[EvidenceCard]:
    """Build up to six evidence cards, one per type, only where evidence exists.

    ``dna`` is accepted for context (e.g. future weighting); current cards derive
    their explanation from each hit's ``dna_keys`` directly.
    """
    cards: List[EvidenceCard] = []

    # --- audio: Audio-source hits, else a transcript snippet ---------------- #
    audio_hits = _hits_by_source(bundle, "Audio")
    if audio_hits:
        cards.append(
            _card(
                "audio",
                audio_hits,
                fragment=_best_fragment(audio_hits),
                confidence=_max_conf(audio_hits),
            )
        )
    elif bundle.transcript.full_text.strip():
        snippet = bundle.transcript.full_text.strip()
        cards.append(
            EvidenceCard(
                type="audio",
                title=TITLES["audio"],
                confidence=clamp_score(40),
                fragment=_quote(snippet),
                explanation="Расшифровка речи доступна для анализа, но явных "
                            "мошеннических фраз в звуке не выделено.",
                timestamp=mmss(bundle.transcript.segments[0].start_s)
                if bundle.transcript.segments else None,
                findings=[],
            )
        )

    # --- ocr: OCR-source hits, else raw ocr_hits ---------------------------- #
    ocr_hits = _hits_by_source(bundle, "OCR")
    if ocr_hits:
        cards.append(
            _card(
                "ocr",
                ocr_hits,
                fragment=_best_fragment(ocr_hits),
                confidence=_max_conf(ocr_hits),
            )
        )
    elif bundle.ocr_hits:
        best = max(bundle.ocr_hits, key=lambda o: o.confidence)
        cards.append(
            EvidenceCard(
                type="ocr",
                title=TITLES["ocr"],
                confidence=clamp_score(best.confidence),
                fragment=_quote(best.text),
                explanation=DEFAULT_EXPLANATIONS["ocr"],
                timestamp=mmss(best.time_s),
                findings=[],
            )
        )

    # --- visual: CLIP visual hits ------------------------------------------- #
    if bundle.visual_hits:
        best_v = max(bundle.visual_hits, key=lambda v: v.score)
        # Use Visual-source analyzer hits for tags/explanation if present.
        visual_src_hits = _hits_by_source(bundle, "Visual")
        labels = []
        for v in sorted(bundle.visual_hits, key=lambda v: v.score, reverse=True):
            lbl = (v.label_ru or v.label or "").strip()
            if lbl and lbl not in labels:
                labels.append(lbl)
            if len(labels) >= MAX_FINDINGS:
                break
        explanation = _explanation("visual", visual_src_hits) \
            if visual_src_hits else DEFAULT_EXPLANATIONS["visual"]
        cards.append(
            EvidenceCard(
                type="visual",
                title=TITLES["visual"],
                confidence=clamp_score(best_v.score * 100.0),
                fragment=_quote(best_v.label_ru or best_v.label),
                explanation=explanation,
                timestamp=mmss(best_v.time_s),
                findings=labels[:MAX_FINDINGS],
            )
        )

    # --- metadata: Metadata-source hits (hashtags / description) ------------ #
    meta_hits = _hits_by_source(bundle, "Metadata")
    if meta_hits:
        extra = list(bundle.media.hashtags)[:MAX_FINDINGS]
        cards.append(
            _card(
                "metadata",
                meta_hits,
                fragment=_best_fragment(meta_hits),
                confidence=_max_conf(meta_hits),
                extra_findings=extra,
            )
        )

    # --- links: external links / contacts ----------------------------------- #
    if bundle.link_hits:
        values: List[str] = []
        for lk in bundle.link_hits:
            v = (lk.value or "").strip()
            if v and v not in values:
                values.append(v)
        fragment = _quote("; ".join(values)) if values else ""
        # Links feed the messenger funnel; reuse Behavior/Metadata hits for conf.
        link_related = [
            h for h in bundle.hits
            if "messenger" in h.dna_keys or "referral" in h.dna_keys
        ]
        confidence = _max_conf(link_related) if link_related else 70.0
        cards.append(
            EvidenceCard(
                type="links",
                title=TITLES["links"],
                confidence=clamp_score(confidence),
                fragment=fragment,
                explanation=_explanation("links", link_related)
                if link_related else DEFAULT_EXPLANATIONS["links"],
                timestamp=_strongest_timestamp(link_related),
                findings=values[:MAX_FINDINGS],
            )
        )

    # --- engagement: Behavior-source hits ----------------------------------- #
    behavior_hits = _hits_by_source(bundle, "Behavior")
    if behavior_hits:
        cards.append(
            _card(
                "engagement",
                behavior_hits,
                fragment=_best_fragment(behavior_hits),
                confidence=_max_conf(behavior_hits),
            )
        )

    return cards


__all__ = ["build_evidence", "mmss"]
