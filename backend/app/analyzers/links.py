"""Link / contact extraction lane.

Pure-stdlib analyzer: scans metadata, transcript and OCR text for external
links and contact handles (Telegram, WhatsApp, phone numbers, promo codes,
short-link redirectors). It both *fills* ``bundle.link_hits`` (consumed by the
behavior lane and scoring) and *returns* ``Metadata``-source :class:`AnalyzerHit`
records for the timeline / evidence builders.

No optional dependencies — always runs, never raises.
"""
from __future__ import annotations

from typing import List, Tuple

from app.config import clamp_score
from app.pipeline.contracts import AnalyzerHit, LinkHit, SignalBundle
from app.scoring.lexicons import LINK_PATTERNS


# kind -> (ScamDNA dimensions it feeds, confidence, RU signal prefix)
_KIND_META = {
    "telegram": (["messenger"], 78, "Внешний контакт в Telegram"),
    "whatsapp": (["messenger"], 80, "Внешний контакт в WhatsApp"),
    "phone": (["messenger"], 60, "Контактный номер телефона"),
    "promocode": (["referral"], 82, "Промокод / реферальный код"),
    "url": (["messenger"], 70, "Внешняя короткая ссылка"),
    "handle": (["messenger"], 60, "Внешний контакт / хэндл"),
}


def _gather_sources(bundle: SignalBundle) -> List[Tuple[str, str]]:
    """Return (source_field, text) pairs to scan, skipping empties."""
    m = bundle.media
    meta_parts = [m.title, m.description, " ".join(m.hashtags)]
    meta_text = "\n".join(p for p in meta_parts if p)
    ocr_text = " ".join(h.text for h in bundle.ocr_hits if h.text)
    sources = [
        ("description", meta_text),
        ("transcript", bundle.transcript.full_text or ""),
        ("ocr", ocr_text),
    ]
    return [(field, text) for field, text in sources if text.strip()]


def extract_links(bundle: SignalBundle) -> list[AnalyzerHit]:
    """Extract external links/contacts.

    Fills ``bundle.link_hits`` (deduped by ``(kind, value)``) and returns a list
    of ``Metadata``-source :class:`AnalyzerHit` records describing each unique
    link. Appends ``"links"`` to ``bundle.lanes_run``.
    """
    bundle.lanes_run.append("links")

    sources = _gather_sources(bundle)
    seen: set[Tuple[str, str]] = {(h.kind, h.value) for h in bundle.link_hits}
    hits: list[AnalyzerHit] = []

    for kind, rx in LINK_PATTERNS.items():
        dna_keys, conf, prefix = _KIND_META.get(
            kind, (["messenger"], 60, "Внешний контакт"))
        for source_field, text in sources:
            for m in rx.finditer(text):
                # Prefer the named "v" group; fall back to the whole match.
                value = m.groupdict().get("v") or m.group(0)
                value = value.strip()
                if not value:
                    continue

                dedup = (kind, value.lower())
                if dedup in seen:
                    continue
                seen.add(dedup)

                bundle.link_hits.append(
                    LinkHit(kind=kind, value=value, source_field=source_field))
                hits.append(AnalyzerHit(
                    source="Metadata",
                    confidence=clamp_score(conf),
                    signal=f"{prefix}: {value}",
                    fragment=value,
                    time_s=None,
                    dna_keys=list(dna_keys),
                    tags=[value],
                ))

    return hits
