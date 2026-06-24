"""Configuration for the app.ml model lifecycle (data → train → calibrate → serve).

All knobs overridable via ``AMW_ML_*`` env vars so training/serving can be tuned
without code edits. The defaults are sized to TRAIN ON CPU IN UNDER A MINUTE on a
few thousand synthetic + weak-labeled examples, while remaining a real learned
multi-task model that generalizes beyond the lexicons (char-n-gram hashing makes
it robust to obfuscation like "г@рантир0ванный д0ход").
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import BACKEND_DIR

# Where trained artifacts + metadata + the active-version pointer live.
ML_DIR = Path(os.getenv("AMW_ML_DIR", BACKEND_DIR / "models_store"))
ML_DIR.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    return int(_num(name, default))


@dataclass(frozen=True)
class MLConfig:
    # --- Serving / integration ----------------------------------------- #
    enable: bool = _flag("AMW_ML_ENABLE", True)        # use model when present
    # How the model score combines with the rule score in the orchestrator:
    #   "model" = model only, "rules" = rules only, "blend" = convex blend.
    blend: str = os.getenv("AMW_ML_BLEND", "blend")
    blend_alpha: float = _num("AMW_ML_BLEND_ALPHA", 0.6)  # weight of MODEL in blend

    model_path: str = str(ML_DIR / "risk_model.npz")
    meta_path: str = str(ML_DIR / "risk_model.json")
    active_pointer: str = str(ML_DIR / "ACTIVE.json")

    # --- Featurization ------------------------------------------------- #
    # Dense hashing dimension (feature-hashing / "hashing trick"). Kept modest so
    # the dense NumPy MLP trains fast; the numeric feature block is concatenated.
    hash_dim: int = _int("AMW_ML_HASH_DIM", 4096)
    char_ngram_min: int = _int("AMW_ML_CHAR_MIN", 3)
    char_ngram_max: int = _int("AMW_ML_CHAR_MAX", 5)
    word_ngram_max: int = _int("AMW_ML_WORD_MAX", 2)
    numeric_dim: int = 32                              # engineered numeric features

    # --- Model --------------------------------------------------------- #
    hidden: int = _int("AMW_ML_HIDDEN", 256)
    dropout: float = _num("AMW_ML_DROPOUT", 0.10)

    # --- Training ------------------------------------------------------ #
    epochs: int = _int("AMW_ML_EPOCHS", 12)
    batch_size: int = _int("AMW_ML_BATCH", 64)
    lr: float = _num("AMW_ML_LR", 0.01)
    l2: float = _num("AMW_ML_L2", 1e-5)
    risk_loss_weight: float = _num("AMW_ML_W_RISK", 1.0)
    dim_loss_weight: float = _num("AMW_ML_W_DIM", 0.5)
    cat_loss_weight: float = _num("AMW_ML_W_CAT", 0.5)
    seed: int = _int("AMW_ML_SEED", 1337)

    # --- Data ---------------------------------------------------------- #
    synth_size: int = _int("AMW_ML_SYNTH", 4000)
    val_frac: float = _num("AMW_ML_VAL_FRAC", 0.15)

    # --- Calibration / uncertainty (active learning) ------------------- #
    calibration: str = os.getenv("AMW_ML_CALIB", "temperature")  # temperature|isotonic|none
    uncertain_low: float = _num("AMW_ML_UNC_LOW", 0.40)
    uncertain_high: float = _num("AMW_ML_UNC_HIGH", 0.60)

    version: str = os.getenv("AMW_ML_VERSION", "amw-risk-0.1.0")


ml_config = MLConfig()
