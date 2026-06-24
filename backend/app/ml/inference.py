"""Serving firewall for the custom risk model (app.ml).

Thin, defensive bridge between the rule-engine orchestrator and the learned
:class:`~app.ml.types.RiskModel`. Its single job is to score a
:class:`~app.pipeline.contracts.SignalBundle` *if and only if* a model is
enabled and available — and to **degrade silently to ``None``** in every other
case so the engine always has its rule-based fallback (lite mode).

Firewall guarantees (non-negotiable, see DESIGN §14):

* ``numpy`` / ``model_np`` / ``featurize`` are imported **lazily** here so this
  module stays importable even when numpy is absent — the orchestrator can
  ``from app.ml.inference import score_bundle`` in a pure-Python deployment.
* :func:`score_bundle` **never raises**: any failure (missing artifact, import
  error, malformed bundle) is caught and turned into ``None``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.ml.config import ml_config

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime numpy import
    from app.ml.types import Prediction
    from app.pipeline.contracts import SignalBundle


def score_bundle(
    bundle: "SignalBundle", kb_similarity: float = 0.0
) -> Optional["Prediction"]:
    """Score a :class:`SignalBundle` with the active learned risk model.

    Returns a :class:`~app.ml.types.Prediction` when the model is enabled and an
    active artifact loads, otherwise ``None`` — signalling the orchestrator to
    fall back to the rule engine. ``kb_similarity`` (0..1, caller-supplied since
    :func:`featurize.extract` leaves it 0) is written onto the extracted
    features before prediction.

    Guaranteed never to raise: every failure path returns ``None``.
    """
    if not ml_config.enable:
        return None

    try:
        # Lazy imports keep this module (and the engine) usable without numpy:
        # both registry.load_active and featurize pull numpy in only on demand.
        from app.ml import featurize
        from app.ml.registry import load_active

        model = load_active()
        if model is None:
            return None

        features = featurize.extract(bundle)
        features.kb_similarity = float(max(0.0, min(1.0, kb_similarity)))
        return model.predict(features)
    except Exception:
        # Serving firewall: any error degrades to the rule-based fallback.
        return None
