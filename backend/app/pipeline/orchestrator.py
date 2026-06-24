"""Tiered-cascade orchestrator — the single entry point that turns a
:class:`MediaInput` into a fully-populated :class:`AnalysisResponse`.

The cascade is deliberately cheap-first:

  1. CHEAP LANE (always): link extraction, lexical text signals and behavioral
     pattern detection run over whatever metadata / captions are supplied. A
     preliminary ScamDNA + risk score is computed from these alone.
  2. ESCALATION: only when there is analyzable uploaded media AND the cheap
     score is non-trivial (or the supplied text was too thin to judge) do we pay
     for the heavy lanes — probe, audio extraction, ASR, keyframes, OCR, vision.
     Afterwards the cheap lanes are RE-RUN so the freshly mined transcript / OCR
     text is also scored, and ``bundle.hits`` is rebuilt from scratch to avoid
     duplicates.
  3. SCORING: ScamDNA -> knowledge-base boost of the "reused" dimension ->
     weighted risk breakdown -> category -> timeline -> evidence -> connections.
  4. ASSEMBLY: a ``CaseResult`` mirroring the frontend ``DemoCase`` shape, plus
     an ``AnalysisMeta`` telemetry envelope.

:func:`analyze` NEVER raises: any internal failure still yields a valid minimal
``CaseResult`` so the API contract holds. Only Python stdlib is imported at
module top level; every leaf module it calls is itself defensive.
"""
from __future__ import annotations

import datetime
import time
import uuid
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from app import __version__
from app.api.schemas import (
    AnalysisMeta,
    AnalysisResponse,
    AnalyzeUrlRequest,
    CaseResult,
    ComponentBreakdown,
    Connections,
    ScamDNADimension,
)
from app.config import clamp_score, risk_level, settings
from app.models import registry
from app.pipeline.contracts import MediaInput, RiskBreakdown, SignalBundle

# Cheap lanes — always available, pure stdlib.
from app.analyzers import behavior, links, text_signals

# Heavy lanes — each is itself defensive and degrades gracefully.
from app.analyzers import asr, ocr, vision
from app.pipeline import audio as audio_lane
from app.pipeline import keyframes as keyframe_lane
from app.pipeline import media_probe

# Scoring.
from app.scoring import category as category_mod
from app.scoring import connections as connections_mod
from app.scoring import evidence as evidence_mod
from app.scoring import risk_score as risk_score_mod
from app.scoring import scam_dna as scam_dna_mod
from app.scoring import timeline as timeline_mod

# Persistence + knowledge base.
from app.store import db
from app.store.knowledge_base import KB

# Platforms the wire contract accepts (schemas.Platform literal).
_VALID_PLATFORMS = {"Instagram", "TikTok", "YouTube", "Telegram", "VK"}

# Below this much metadata text the cheap lane cannot reasonably judge an upload,
# so we escalate to the heavy lanes even if the preliminary score is low.
_THIN_TEXT_CHARS = 40


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def analyze(media: MediaInput) -> AnalysisResponse:
    """Run the tiered cascade over ``media`` and return the full response.

    Never raises — on any internal error it returns a valid minimal case so the
    HTTP layer always has a well-formed body to serialize.
    """
    t0 = time.perf_counter()
    bundle = SignalBundle(media=media)

    try:
        return _analyze(bundle, t0)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        return _fallback_response(bundle, t0, exc)


# --------------------------------------------------------------------------- #
# Cascade implementation
# --------------------------------------------------------------------------- #
def _analyze(bundle: SignalBundle, t0: float) -> AnalysisResponse:
    media = bundle.media

    # --- (2) CHEAP LANE — always ---------------------------------------- #
    _run_cheap_lanes(bundle)

    # Preliminary scoring from cheap signals only.
    prelim_dna = scam_dna_mod.compute_scam_dna(bundle)
    prelim_breakdown = risk_score_mod.compute_risk_score(bundle, prelim_dna)

    # --- (3) ESCALATE to heavy lanes when warranted --------------------- #
    if _should_escalate(media, prelim_breakdown):
        _run_heavy_lanes(bundle)
        # Re-mine everything (now including transcript + OCR text). Rebuild
        # bundle.hits AND link_hits from scratch so cheap-lane results are not
        # duplicated and link-derived Metadata hits are re-emitted (extract_links
        # skips links already present in bundle.link_hits).
        bundle.hits = []
        bundle.link_hits = []
        _run_cheap_lanes(bundle)

    # --- (4) SCORING ---------------------------------------------------- #
    dna = scam_dna_mod.compute_scam_dna(bundle)

    # Knowledge-base similarity boosts the "reused" dimension.
    kb_sim = _kb_similarity(bundle)
    _boost_reused(dna, kb_sim)

    breakdown = risk_score_mod.compute_risk_score(bundle, dna)
    cat, cat_ru = category_mod.classify_category(bundle, dna, breakdown)

    # Learned-model lane — blends with / overrides the rule score when a trained
    # model is present; otherwise a no-op (pure rule path). Mutates dna in place.
    final_score, final_level, ml_notes = _apply_model(bundle, dna, breakdown, kb_sim)

    timeline = timeline_mod.build_timeline(bundle)
    evidence = evidence_mod.build_evidence(bundle, dna)
    connections = connections_mod.build_connections(bundle, KB, final_score)

    # --- (5) ASSEMBLE the CaseResult ------------------------------------ #
    case = CaseResult(
        id="case-" + uuid.uuid4().hex[:8],
        title=_title(media),
        platform=_platform(media),
        duration=_duration(bundle),
        riskScore=final_score,
        riskLevel=final_level,  # type: ignore[arg-type]
        category=cat,
        categoryRu=cat_ru,
        status="new",
        uploadDate=datetime.date.today().isoformat(),
        description=_describe(media, breakdown, dna, kb_sim),
        mainReason=breakdown.main_reason,
        hashtags=_hashtags(media),
        scamDNA=dna,
        timeline=timeline,
        evidenceCards=evidence,
        connections=connections,
    )

    # --- (6) META telemetry --------------------------------------------- #
    meta = AnalysisMeta(
        engineMode=registry.engine_mode(),  # type: ignore[arg-type]
        degraded=bundle.degradation.fell_back(),
        lanesRun=sorted(set(bundle.lanes_run)),
        elapsedMs=int((time.perf_counter() - t0) * 1000),
        components=ComponentBreakdown(
            text_speech=breakdown.text_speech,
            visual=breakdown.visual,
            metadata_links=breakdown.metadata_links,
            behavior=breakdown.behavior,
            db_similarity=breakdown.db_similarity,
        ),
        notes=list(bundle.degradation.notes) + ml_notes,
    )

    # --- (7) PERSIST (best-effort) -------------------------------------- #
    try:
        db.save_case(case, meta)
    except Exception:  # noqa: BLE001 - persistence is never fatal
        pass

    return AnalysisResponse(case=case, meta=meta)


def _run_cheap_lanes(bundle: SignalBundle) -> None:
    """Links -> text -> behavior. Each extends ``bundle.hits`` in declared order.

    ``links.extract_links`` also fills ``bundle.link_hits`` (consumed by behavior
    and the connection graph), so it must run first.
    """
    bundle.hits.extend(links.extract_links(bundle))
    bundle.hits.extend(text_signals.analyze_text(bundle))
    bundle.hits.extend(behavior.analyze_behavior(bundle))


def _run_heavy_lanes(bundle: SignalBundle) -> None:
    """Probe -> audio -> ASR -> keyframes -> OCR -> vision (all defensive)."""
    media = bundle.media

    probe = media_probe.probe(media)
    bundle.probe = probe

    # Audio + speech-to-text.
    track = audio_lane.extract_audio(media, probe)
    asr.run_asr(bundle, track)

    # Keyframes -> OCR + vision.
    keyframes = keyframe_lane.extract_keyframes(media, probe)
    if not keyframes and probe.ok and probe.has_video:
        # Probe says there is video but we could not sample any frames.
        bundle.degradation.media = True
        bundle.degradation.notes.append(
            "Не удалось извлечь кадры из видео — визуальные дорожки пропущены."
        )
    ocr.run_ocr(bundle, keyframes)
    vision.run_vision(bundle, keyframes)


def _should_escalate(media: MediaInput, prelim: RiskBreakdown) -> bool:
    """Escalate only for analyzable uploads that are either non-trivially risky
    or backed by too little text to judge cheaply."""
    if media.source_type != "upload" or not media.path:
        return False
    thin_text = len(_metadata_text(media)) < _THIN_TEXT_CHARS
    return prelim.score >= settings.cascade_short_circuit_below or thin_text


# --------------------------------------------------------------------------- #
# Knowledge-base / "reused" boost
# --------------------------------------------------------------------------- #
def _kb_similarity(bundle: SignalBundle) -> dict:
    """Best-effort KB similarity; never raises out of analysis."""
    try:
        return KB.similarity(bundle)
    except Exception:  # noqa: BLE001
        return {"score": 0, "cluster_size": 1, "description": "", "related": []}


def _boost_reused(dna: List[ScamDNADimension], kb_sim: dict) -> None:
    """Raise the ``reused`` dimension toward the KB similarity score in place.

    The risk-score scorer derives ``db_similarity`` straight from the ``reused``
    dimension, so boosting it here is what makes the KB match flow into the
    overall score and the ``metadata_links`` component is unaffected.
    """
    try:
        kb_score = clamp_score(float(kb_sim.get("score", 0) or 0))
    except (TypeError, ValueError):
        kb_score = 0
    if kb_score <= 0:
        return
    for dim in dna:
        if dim.key == "reused":
            if kb_score > dim.value:
                dim.value = kb_score
                desc = str(kb_sim.get("description", "")).strip()
                if desc:
                    dim.description = desc
            break


def _apply_model(
    bundle: SignalBundle,
    dna: List[ScamDNADimension],
    breakdown: RiskBreakdown,
    kb_sim: dict,
) -> Tuple[int, str, List[str]]:
    """Blend the learned model (app.ml) with the rule score — additive & fail-safe.

    Returns ``(final_score, final_level, notes)``. When the model is disabled or
    no trained artifact is available this is a NO-OP returning the rule score
    unchanged, so the engine path stays byte-for-byte the rule-based behaviour.
    When a model is present, the 8 ScamDNA dimension values are also blended
    toward the model in place so the UI reflects it. Never raises.
    """
    rule_score = breakdown.score
    try:
        from app.ml.config import ml_config
        if not ml_config.enable:
            return rule_score, breakdown.level, []
        from app.ml.inference import score_bundle as _ml_score
        try:
            kb_score = float(kb_sim.get("score", 0) or 0) / 100.0
        except (TypeError, ValueError):
            kb_score = 0.0
        pred = _ml_score(bundle, kb_similarity=kb_score)
    except Exception:  # noqa: BLE001 - serving firewall: fall back to rules
        return rule_score, breakdown.level, []

    if pred is None:
        return rule_score, breakdown.level, []

    mode = ml_config.blend
    alpha = max(0.0, min(1.0, ml_config.blend_alpha))
    if mode == "model":
        final = float(pred.risk_score)
    elif mode == "rules":
        final = float(rule_score)
    else:  # "blend"
        final = alpha * pred.risk_score + (1.0 - alpha) * rule_score

    if mode != "rules" and pred.dimensions:
        for dim in dna:
            mv = pred.dimensions.get(dim.key)
            if mv is not None:
                dim.value = clamp_score(alpha * float(mv) + (1.0 - alpha) * dim.value)

    notes = [
        f"ml: {pred.model_version} (blend={mode}, alpha={alpha:.2f}, "
        f"model={pred.risk_score}, rules={rule_score}, conf={pred.confidence:.2f})"
    ]
    if getattr(pred, "uncertain", False):
        notes.append("ml: низкая уверенность модели — рекомендуется ручная проверка.")
    return clamp_score(final), risk_level(final), notes


# --------------------------------------------------------------------------- #
# CaseResult field derivation
# --------------------------------------------------------------------------- #
def _metadata_text(media: MediaInput) -> str:
    parts = [media.title, media.description, " ".join(media.hashtags)]
    if media.provided_transcript:
        parts.append(media.provided_transcript)
    return " ".join(p for p in parts if p).strip()


def _title(media: MediaInput) -> str:
    """Case title: explicit title, else a name derived from the filename."""
    if media.title and media.title.strip():
        return media.title.strip()
    fname = (media.filename or "").strip()
    if fname:
        stem = fname.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
        stem = stem.replace("_", " ").replace("-", " ").strip()
        if stem:
            return stem[:80]
    return "Загруженное видео"


def _platform(media: MediaInput) -> str:
    """Resolve a valid Platform: explicit -> inferred from URL -> Instagram."""
    if media.platform and media.platform in _VALID_PLATFORMS:
        return media.platform
    inferred = _platform_from_url(media.url)
    return inferred or "Instagram"


def _platform_from_url(url: Optional[str]) -> Optional[str]:
    """Map a URL's domain to a known Platform literal, if recognizable."""
    if not url:
        return None
    try:
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, AttributeError):
        return None
    if not host:
        # Maybe a bare domain without a scheme.
        host = url.lower()
    table = (
        ("instagram", "Instagram"),
        ("tiktok", "TikTok"),
        ("youtube", "YouTube"),
        ("youtu.be", "YouTube"),
        ("t.me", "Telegram"),
        ("telegram", "Telegram"),
        ("vk.com", "VK"),
        ("vk.ru", "VK"),
        ("vkvideo", "VK"),
    )
    for needle, platform in table:
        if needle in host:
            return platform
    return None


def _duration(bundle: SignalBundle) -> str:
    """``M:SS`` from the probe duration, else the last keyframe time, else 0:00."""
    seconds = 0.0
    if bundle.probe.ok and bundle.probe.duration_s > 0:
        seconds = bundle.probe.duration_s
    elif bundle.transcript.segments:
        seconds = max(s.end_s for s in bundle.transcript.segments)
    elif bundle.visual_hits or bundle.ocr_hits:
        times = [h.time_s for h in bundle.visual_hits]
        times += [h.time_s for h in bundle.ocr_hits]
        if times:
            seconds = max(times)
    return mmss(seconds)


def _hashtags(media: MediaInput) -> List[str]:
    """Explicit hashtags, else hashtags scraped from the textual metadata."""
    if media.hashtags:
        out: List[str] = []
        seen = set()
        for tag in media.hashtags:
            t = ("#" + tag.lstrip("#")).strip()
            low = t.lower()
            if len(t) > 1 and low not in seen:
                seen.add(low)
                out.append(t)
        if out:
            return out
    return _extract_hashtags_from_text(_metadata_text(media))


def _extract_hashtags_from_text(text: str) -> List[str]:
    """Pull ``#tag`` tokens out of free text, order-preserving and deduped."""
    out: List[str] = []
    seen = set()
    for token in text.split():
        token = token.strip(".,!?;:()[]{}\"'«»")
        if token.startswith("#") and len(token) > 1:
            low = token.lower()
            if low not in seen:
                seen.add(low)
                out.append(token)
    return out[:12]


def _describe(
    media: MediaInput,
    breakdown: RiskBreakdown,
    dna: List[ScamDNADimension],
    kb_sim: dict,
) -> str:
    """Generate a short Russian summary of the main findings."""
    level_ru = {
        "critical": "критический",
        "high": "высокий",
        "medium": "средний",
        "low": "низкий",
    }.get(breakdown.level, "неопределённый")

    sentences = [
        f"Уровень риска: {level_ru} ({breakdown.score}/100). {breakdown.main_reason}."
    ]

    # Surface the two strongest ScamDNA dimensions, if any are meaningful.
    top = sorted(dna, key=lambda d: d.value, reverse=True)
    strong = [d for d in top if d.value >= 40][:2]
    if strong:
        names = ", ".join(f"{d.nameRu} ({d.value})" for d in strong)
        sentences.append(f"Ведущие признаки: {names}.")

    try:
        kb_score = int(kb_sim.get("score", 0) or 0)
    except (TypeError, ValueError):
        kb_score = 0
    if kb_score >= 30 and kb_sim.get("description"):
        sentences.append(
            f"Совпадение с базой известных схем: {kb_score}%."
        )

    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mmss(seconds: float) -> str:
    """Format a number of seconds as ``"M:SS"`` (matches the DemoCase shape)."""
    total = int(round(max(0.0, seconds)))
    return f"{total // 60}:{total % 60:02d}"


def make_media_from_upload(
    path: str,
    filename: str,
    size: int,
    title: str = "",
    platform: Optional[str] = None,
    description: str = "",
    hashtags: Optional[List[str]] = None,
    provided_transcript: Optional[str] = None,
) -> MediaInput:
    """Build a :class:`MediaInput` for an uploaded file."""
    plat = platform if platform in _VALID_PLATFORMS else None
    return MediaInput(
        source_type="upload",
        path=path,
        filename=filename,
        size_bytes=int(size or 0),
        platform=plat,  # type: ignore[arg-type]
        title=title or "",
        description=description or "",
        hashtags=list(hashtags or []),
        provided_transcript=provided_transcript,
    )


def make_media_from_url(req: AnalyzeUrlRequest) -> MediaInput:
    """Build a :class:`MediaInput` from an analyze-by-reference request.

    No heavy download happens: ``source_type`` is ``"url"`` when a URL is given
    (so a domain-based platform can be inferred) and ``"text"`` otherwise. Only
    the supplied lightweight metadata / captions are analyzed.
    """
    url = (req.url or "").strip() or None
    source_type = "url" if url else "text"
    plat = req.platform if req.platform in _VALID_PLATFORMS else None
    return MediaInput(
        source_type=source_type,  # type: ignore[arg-type]
        url=url,
        platform=plat,  # type: ignore[arg-type]
        title=req.title or "",
        description=req.description or "",
        hashtags=list(req.hashtags or []),
        provided_transcript=req.transcript,
    )


# --------------------------------------------------------------------------- #
# Last-resort fallback
# --------------------------------------------------------------------------- #
def _fallback_response(
    bundle: SignalBundle, t0: float, exc: Exception
) -> AnalysisResponse:
    """Construct a valid minimal response when the cascade itself fails."""
    media = bundle.media
    note = f"Внутренняя ошибка анализа ({type(exc).__name__}); выдан минимальный отчёт."
    case = CaseResult(
        id="case-" + uuid.uuid4().hex[:8],
        title=_title(media),
        platform=_platform(media),
        duration=_duration(bundle),
        riskScore=0,
        riskLevel=risk_level(0),  # type: ignore[arg-type]
        category="unknown",
        categoryRu="Не удалось определить",
        status="new",
        uploadDate=datetime.date.today().isoformat(),
        description="Анализ не завершился из-за внутренней ошибки. "
                    "Требуется повторная проверка.",
        mainReason="Анализ не завершён",
        hashtags=_hashtags(media),
        scamDNA=[],
        timeline=[],
        evidenceCards=[],
        connections=Connections(),
    )
    meta = AnalysisMeta(
        engineMode=registry.engine_mode(),  # type: ignore[arg-type]
        degraded=bundle.degradation.fell_back(),
        lanesRun=sorted(set(bundle.lanes_run)),
        elapsedMs=int((time.perf_counter() - t0) * 1000),
        components=ComponentBreakdown(),
        notes=list(bundle.degradation.notes) + [note],
    )
    try:
        db.save_case(case, meta)
    except Exception:  # noqa: BLE001
        pass
    return AnalysisResponse(case=case, meta=meta)
