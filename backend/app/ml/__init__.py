"""app.ml — AMW custom risk model.

A purpose-built, multi-task, calibrated and explainable scam-risk model for
short social-media videos, trained by WEAK-SUPERVISION DISTILLATION from the
rule engine plus synthetic/adversarial data, and served with a portable
zero-heavy-dependency inference path so it runs in the engine's lite mode.

This package is ADDITIVE: when no trained model (or numpy) is present the engine
falls back to the rule-based scorer. See app/ml/README.md and DESIGN.md.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
