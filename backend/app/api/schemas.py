"""Pydantic API contract.

These models mirror ``src/types/index.ts`` (the ``DemoCase`` graph) FIELD FOR
FIELD, including camelCase names, so the JSON the engine emits is a drop-in
replacement for the frontend's mock ``DemoCase`` objects. Do not rename fields.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# --- Literal unions (mirror the TS string-literal types) ------------------- #
RiskLevel = Literal["low", "medium", "high", "critical"]
SignalSource = Literal["OCR", "Audio", "Visual", "Metadata", "Behavior"]
CaseStatus = Literal["new", "reviewing", "confirmed", "false_positive", "archived"]
EvidenceType = Literal["audio", "ocr", "visual", "metadata", "links", "engagement"]
Platform = Literal["Instagram", "TikTok", "YouTube", "Telegram", "VK"]
NodeType = Literal["video", "account", "hashtag", "telegram"]
EdgeType = Literal["account", "telegram", "hashtag", "related", "pattern"]


class ScamDNADimension(BaseModel):
    key: str
    name: str
    nameRu: str
    value: int = Field(ge=0, le=100)
    description: str


class TimelineEvent(BaseModel):
    id: str
    time: str                      # "MM:SS"
    timeSeconds: int
    source: SignalSource
    signal: str
    confidence: int = Field(ge=0, le=100)
    severity: RiskLevel


class EvidenceCard(BaseModel):
    type: EvidenceType
    title: str
    confidence: int = Field(ge=0, le=100)
    fragment: str
    explanation: str
    timestamp: Optional[str] = None
    findings: List[str] = Field(default_factory=list)


class ConnectionNode(BaseModel):
    id: str
    type: NodeType
    label: str
    riskScore: Optional[int] = None
    x: float
    y: float


class ConnectionEdge(BaseModel):
    source: str
    target: str
    type: EdgeType


class Connections(BaseModel):
    nodes: List[ConnectionNode] = Field(default_factory=list)
    edges: List[ConnectionEdge] = Field(default_factory=list)
    clusterSize: int = 1
    clusterDescription: str = ""


class CaseResult(BaseModel):
    """Exact analog of the TS ``DemoCase`` interface."""
    id: str
    title: str
    platform: Platform
    duration: str                  # "M:SS"
    riskScore: int = Field(ge=0, le=100)
    riskLevel: RiskLevel
    category: str
    categoryRu: str
    status: CaseStatus = "new"
    uploadDate: str                # "YYYY-MM-DD"
    description: str
    mainReason: str
    hashtags: List[str] = Field(default_factory=list)
    scamDNA: List[ScamDNADimension] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    evidenceCards: List[EvidenceCard] = Field(default_factory=list)
    connections: Connections = Field(default_factory=Connections)


# --------------------------------------------------------------------------- #
# Request / response envelopes
# --------------------------------------------------------------------------- #
class AnalyzeUrlRequest(BaseModel):
    """Analyze by reference WITHOUT downloading the heavy video: the engine pulls
    only lightweight signals (metadata, captions, thumbnail) where available."""
    url: Optional[str] = None
    # Free string here (not the strict Platform literal): the orchestrator
    # validates/normalizes it, so an unknown value falls back instead of 422-ing.
    platform: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    # Optional pre-extracted text (e.g. platform captions) to skip ASR entirely.
    transcript: Optional[str] = None

    @field_validator("hashtags", mode="before")
    @classmethod
    def _coerce_hashtags(cls, v: object) -> List[str]:
        """Accept either a list or a comma/space-separated string from clients."""
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for chunk in v.split(",") for p in chunk.split()]
            return [p for p in parts if p]
        return list(v)  # type: ignore[arg-type]


class ComponentBreakdown(BaseModel):
    """The five weighted components behind the Risk Score (for explainability)."""
    text_speech: int = 0
    visual: int = 0
    metadata_links: int = 0
    behavior: int = 0
    db_similarity: int = 0


class AnalysisMeta(BaseModel):
    """Engine telemetry — NOT part of the DemoCase shape; lives alongside it."""
    engineMode: Literal["full", "hybrid", "lite"] = "lite"
    degraded: List[str] = Field(default_factory=list)     # lanes that fell back
    lanesRun: List[str] = Field(default_factory=list)     # lanes actually executed
    elapsedMs: int = 0
    components: ComponentBreakdown = Field(default_factory=ComponentBreakdown)
    notes: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    case: CaseResult
    meta: AnalysisMeta


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engineMode: str
    capabilities: dict
