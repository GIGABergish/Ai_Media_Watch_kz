"""FIXED public contracts for the custom risk model (app.ml).

Every ml/ module implements against these dataclasses / Protocols. Implementations
MAY append new OPTIONAL fields, but MUST NOT rename or remove existing ones — the
serialized model artifact, the training pipeline and the orchestrator all depend
on this shape. Mirrors the 8 ScamDNA dimensions of the wider system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

# The 8 ScamDNA dimensions the model predicts as auxiliary (multi-task) heads.
# MUST stay in this order and match app.scoring.lexicons.DIMENSION_KEYS.
DIMENSION_KEYS: List[str] = [
    "profit", "urgency", "gambling", "referral",
    "messenger", "visual", "reused", "hashtags",
]

# Coarse category labels the classifier head emits (machine keys).
CATEGORY_KEYS: List[str] = [
    "illegal_gambling", "financial_pyramid", "investment_scam",
    "educational", "educational_antifraud", "suspicious_other", "no_violation",
]


@dataclass
class RawFeatures:
    """Domain features extracted from a SignalBundle (pre-vectorization).

    This is the single input type to the model: ``featurize.extract(bundle)``
    produces it, and ``featurize.vectorize(rf)`` turns it into the numeric array
    the model consumes. ``text`` carries the combined transcript + OCR + metadata.
    """
    text: str = ""
    title: str = ""
    description: str = ""
    hashtags: List[str] = field(default_factory=list)
    link_counts: Dict[str, int] = field(default_factory=dict)     # kind -> count
    behavior_flags: Dict[str, float] = field(default_factory=dict)
    visual_scores: Dict[str, float] = field(default_factory=dict)  # dna_key -> 0..1
    kb_similarity: float = 0.0          # 0..1
    duration_s: float = 0.0
    num_segments: int = 0
    lang_hint: str = ""                 # "ru" | "kz" | "en" | "mixed" | ""


@dataclass
class Label:
    """Training target — from weak supervision (rule engine), synthetic, or gold."""
    risk: float = 0.0                                   # 0..1
    dimensions: Dict[str, float] = field(default_factory=dict)  # key -> 0..1
    category: str = ""                                  # one of CATEGORY_KEYS
    is_scam: bool = False
    source: str = "weak"                                # weak | synthetic | gold
    weight: float = 1.0                                 # sample weight / teacher confidence


@dataclass
class Example:
    """A single (features, label) training/eval row."""
    id: str
    features: RawFeatures
    label: Optional[Label] = None


@dataclass
class Attribution:
    """One explainability contribution toward the prediction."""
    feature: str                # human-readable feature / token
    weight: float               # signed contribution
    dna_key: str = ""           # ScamDNA dimension it maps to, when applicable


@dataclass
class Prediction:
    """Model output — superset of what the engine needs to build a CaseResult."""
    risk_score: int = 0                                 # 0..100
    risk_prob: float = 0.0                              # calibrated 0..1
    risk_level: str = "low"                            # low|medium|high|critical
    dimensions: Dict[str, int] = field(default_factory=dict)   # key -> 0..100
    category: str = ""
    confidence: float = 0.0                            # 0..1
    uncertain: bool = False                            # flag for human review
    attributions: List[Attribution] = field(default_factory=list)
    model_version: str = ""


@runtime_checkable
class RiskModel(Protocol):
    """The trainable, servable model interface.

    ``load`` is provided as a staticmethod/classmethod by implementations
    (``MyModel.load(path) -> RiskModel``); it is intentionally not in the
    Protocol because Protocols cannot constrain constructors cleanly.
    """
    model_version: str

    def predict(self, features: RawFeatures) -> Prediction: ...
    def predict_batch(self, batch: List[RawFeatures]) -> List[Prediction]: ...
    def save(self, path: str) -> None: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, bundle: object) -> RawFeatures: ...   # bundle: SignalBundle
