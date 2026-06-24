"""Internal pipeline contracts — the dataclasses that flow through the engine
and the function signatures every analyzer / scorer module MUST implement.

Data flow (tiered cascade):

    MediaInput
        │  pipeline.media_probe.probe()            -> ProbeResult
        │  pipeline.audio.extract_audio()          -> Optional[AudioTrack]
        │  pipeline.keyframes.extract_keyframes()  -> list[Keyframe]
        ▼
    SignalBundle  (mutated in place by each analyzer)
        │  analyzers.asr.run_asr()                 -> fills .transcript
        │  analyzers.ocr.run_ocr()                 -> fills .ocr_hits
        │  analyzers.vision.run_vision()           -> fills .visual_hits
        │  analyzers.links.extract_links()         -> fills .link_hits
        │  analyzers.text_signals.analyze_text()   -> returns list[AnalyzerHit]
        │  analyzers.behavior.analyze_behavior()   -> returns list[AnalyzerHit]
        ▼
    scoring.scam_dna.compute_scam_dna()            -> list[ScamDNADimension]
    scoring.risk_score.compute_risk_score()        -> RiskBreakdown
    scoring.category.classify_category()           -> (category, categoryRu)
    scoring.timeline.build_timeline()              -> list[TimelineEvent]
    scoring.evidence.build_evidence()              -> list[EvidenceCard]
    scoring.connections.build_connections()        -> Connections
        ▼
    pipeline.orchestrator.analyze()                -> AnalysisResponse

Every analyzer is DEFENSIVE: if its optional ML dependency is missing it must set
the relevant ``Degradation`` flag on the bundle and return cheap heuristic results
(or empty), never raise. The engine must always produce a full, valid CaseResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

# Re-export the wire enums so internal modules share one definition.
from app.api.schemas import RiskLevel, SignalSource, EvidenceType, Platform

SourceType = Literal["upload", "url", "text"]
LinkKind = Literal["telegram", "whatsapp", "url", "promocode", "phone", "handle"]


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
@dataclass
class MediaInput:
    source_type: SourceType
    # For uploads:
    path: Optional[str] = None
    filename: Optional[str] = None
    size_bytes: int = 0
    # For url / text references (lightweight, no heavy download):
    url: Optional[str] = None
    # Metadata / textual context (always cheap, always analyzed first):
    platform: Optional[Platform] = None
    title: str = ""
    description: str = ""
    hashtags: List[str] = field(default_factory=list)
    # Optionally pre-supplied transcript (e.g. platform captions) -> skips ASR.
    provided_transcript: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction artifacts
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    duration_s: float = 0.0
    has_audio: bool = False
    has_video: bool = False
    width: int = 0
    height: int = 0
    ok: bool = False                 # False when no media could be probed
    error: Optional[str] = None


@dataclass
class Keyframe:
    index: int
    time_s: float
    # PIL.Image.Image when available, else None. Kept as Any to avoid a hard
    # Pillow import in the contract module.
    image: object = None
    path: Optional[str] = None


@dataclass
class AudioTrack:
    path: Optional[str] = None       # 16 kHz mono wav path, when extracted
    sample_rate: int = 16000
    duration_s: float = 0.0
    # numpy float32 array when loaded in-memory, else None.
    samples: object = None


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


@dataclass
class Transcript:
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""
    source: Literal["whisper", "captions", "provided", "none"] = "none"


@dataclass
class OcrHit:
    time_s: float
    text: str
    confidence: float = 0.0          # 0..100


@dataclass
class VisualHit:
    time_s: float
    label: str                       # e.g. "casino slot interface"
    label_ru: str = ""
    score: float = 0.0               # 0..1 (CLIP cosine / softmax prob)


@dataclass
class LinkHit:
    kind: LinkKind
    value: str
    source_field: str = ""           # "description" | "ocr" | "transcript" ...


# --------------------------------------------------------------------------- #
# Generic analyzer signal — the common currency for scoring/timeline/evidence.
# --------------------------------------------------------------------------- #
@dataclass
class AnalyzerHit:
    source: SignalSource             # OCR | Audio | Visual | Metadata | Behavior
    confidence: float                # 0..100
    signal: str                      # human-readable RU description of the hit
    fragment: str = ""               # the matched fragment / quote
    time_s: Optional[float] = None   # when known (drives the timeline)
    dna_keys: List[str] = field(default_factory=list)   # ScamDNA dims it feeds
    category_hints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)       # short finding chips


# --------------------------------------------------------------------------- #
# Degradation tracking — which optional lanes fell back this run.
# --------------------------------------------------------------------------- #
@dataclass
class Degradation:
    asr: bool = False
    ocr: bool = False
    vision: bool = False
    media: bool = False              # no ffmpeg / could not read media
    notes: List[str] = field(default_factory=list)

    def fell_back(self) -> List[str]:
        out = []
        for name in ("asr", "ocr", "vision", "media"):
            if getattr(self, name):
                out.append(name)
        return out


# --------------------------------------------------------------------------- #
# The central mutable bundle carried through the whole pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class SignalBundle:
    media: MediaInput
    probe: ProbeResult = field(default_factory=ProbeResult)
    transcript: Transcript = field(default_factory=Transcript)
    ocr_hits: List[OcrHit] = field(default_factory=list)
    visual_hits: List[VisualHit] = field(default_factory=list)
    link_hits: List[LinkHit] = field(default_factory=list)
    hits: List[AnalyzerHit] = field(default_factory=list)   # text + behavior + …
    degradation: Degradation = field(default_factory=Degradation)
    lanes_run: List[str] = field(default_factory=list)

    # Convenience accessors -------------------------------------------------- #
    def all_text(self) -> str:
        """Every piece of text available for cheap lexical analysis."""
        parts = [
            self.media.title,
            self.media.description,
            " ".join(self.media.hashtags),
            self.transcript.full_text,
            " ".join(h.text for h in self.ocr_hits),
        ]
        return "\n".join(p for p in parts if p)

    def add_hit(self, hit: AnalyzerHit) -> None:
        self.hits.append(hit)


# --------------------------------------------------------------------------- #
# Scoring outputs
# --------------------------------------------------------------------------- #
@dataclass
class RiskBreakdown:
    score: int                       # 0..100 overall
    level: RiskLevel
    text_speech: int = 0             # the five weighted components (0..100 each)
    visual: int = 0
    metadata_links: int = 0
    behavior: int = 0
    db_similarity: int = 0
    main_reason: str = ""


__all__ = [
    "SourceType", "LinkKind", "MediaInput", "ProbeResult", "Keyframe",
    "AudioTrack", "TranscriptSegment", "Transcript", "OcrHit", "VisualHit",
    "LinkHit", "AnalyzerHit", "Degradation", "SignalBundle", "RiskBreakdown",
    "RiskLevel", "SignalSource", "EvidenceType", "Platform",
]
