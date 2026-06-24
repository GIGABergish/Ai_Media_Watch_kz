"""Tests for the custom risk model (app.ml).

A single self-checking, pytest-free smoke test lives here so the whole ML
pipeline (synth -> dataset -> NpRiskModel.fit -> calibrate -> predict ->
save/load) can be validated end-to-end with::

    python -m app.ml.tests.test_pipeline

It runs on a tiny ``MLConfig`` override (few hundred synthetic rows, a handful of
epochs) so it finishes in seconds and stays deterministic.
"""

__all__: list[str] = []
