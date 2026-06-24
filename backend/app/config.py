"""Central configuration — the single source of truth for weights, thresholds,
feature flags and shared level helpers. Every module imports from here so the
scoring stays consistent across the codebase.

All knobs are overridable via environment variables (prefix ``AMW_``) so the
engine can be tuned without touching code. See ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent.parent          # .../backend
DATA_DIR = Path(os.getenv("AMW_DATA_DIR", BACKEND_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("AMW_UPLOAD_DIR", DATA_DIR / "uploads"))
DB_PATH = Path(os.getenv("AMW_DB_PATH", DATA_DIR / "cases.db"))

for _p in (DATA_DIR, UPLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Risk Score formula weights — MUST sum to 1.0.
# Mirrors the "Формула Risk Score" block in the frontend SettingsPage.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskWeights:
    text_speech: float = 0.35       # Текст и речь
    visual: float = 0.25            # Визуальные признаки
    metadata_links: float = 0.15    # Метаданные и ссылки
    behavior: float = 0.15          # Поведенческие паттерны
    db_similarity: float = 0.10     # Похожесть на базу

    def as_dict(self) -> dict:
        return {
            "text_speech": self.text_speech,
            "visual": self.visual,
            "metadata_links": self.metadata_links,
            "behavior": self.behavior,
            "db_similarity": self.db_similarity,
        }


@dataclass(frozen=True)
class Thresholds:
    # Overall risk level cut-offs (riskScore 0..100 -> RiskLevel)
    critical: int = 88
    high: int = 65
    medium: int = 40
    # Per-signal severity cut-offs (confidence 0..100 -> RiskLevel)
    sev_critical: int = 85
    sev_high: int = 65
    sev_medium: int = 45


@dataclass(frozen=True)
class Settings:
    weights: RiskWeights = field(default_factory=RiskWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # --- Cascade control ------------------------------------------------ #
    # The tiered orchestrator escalates to heavier lanes only when the cheap
    # text lane lands inside this ambiguous band (or above the lower bound).
    cascade_escalate_above: float = _num("AMW_ESCALATE_ABOVE", 25.0)
    cascade_short_circuit_below: float = _num("AMW_SHORTCIRCUIT_BELOW", 8.0)

    # --- Optional ML lanes (lazy, degrade gracefully when unavailable) -- #
    enable_asr: bool = _flag("AMW_ENABLE_ASR", True)
    enable_ocr: bool = _flag("AMW_ENABLE_OCR", True)
    enable_vision: bool = _flag("AMW_ENABLE_VISION", True)

    whisper_model: str = os.getenv("AMW_WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("AMW_WHISPER_DEVICE", "cpu")
    clip_model: str = os.getenv("AMW_CLIP_MODEL", "ViT-B-32")
    clip_pretrained: str = os.getenv("AMW_CLIP_PRETRAINED", "openai")
    tesseract_lang: str = os.getenv("AMW_TESSERACT_LANG", "rus+eng")

    # --- Keyframe sampling (efficiency: sample, never full-decode) ------ #
    keyframe_interval_s: float = _num("AMW_KEYFRAME_INTERVAL", 2.0)
    max_keyframes: int = int(_num("AMW_MAX_KEYFRAMES", 24))
    keyframe_max_dim: int = int(_num("AMW_KEYFRAME_MAX_DIM", 720))

    # --- Server -------------------------------------------------------- #
    host: str = os.getenv("AMW_HOST", "127.0.0.1")
    port: int = int(_num("AMW_PORT", 8000))
    max_upload_mb: int = int(_num("AMW_MAX_UPLOAD_MB", 500))

    @property
    def cors_origins(self) -> Tuple[str, ...]:
        env = os.getenv("AMW_CORS_ORIGINS")
        if env:
            return tuple(o.strip() for o in env.split(",") if o.strip())
        return (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        )


settings = Settings()


# --------------------------------------------------------------------------- #
# Shared level helpers — import these everywhere; never re-implement thresholds.
# --------------------------------------------------------------------------- #
def risk_level(score: float) -> str:
    """Map an overall 0..100 risk score to a RiskLevel literal."""
    t = settings.thresholds
    if score >= t.critical:
        return "critical"
    if score >= t.high:
        return "high"
    if score >= t.medium:
        return "medium"
    return "low"


def severity_from_confidence(confidence: float) -> str:
    """Map a per-signal 0..100 confidence to a RiskLevel severity literal."""
    t = settings.thresholds
    if confidence >= t.sev_critical:
        return "critical"
    if confidence >= t.sev_high:
        return "high"
    if confidence >= t.sev_medium:
        return "medium"
    return "low"


def clamp_score(value: float) -> int:
    """Clamp to the inclusive 0..100 integer range used everywhere in the UI."""
    return int(max(0, min(100, round(value))))
