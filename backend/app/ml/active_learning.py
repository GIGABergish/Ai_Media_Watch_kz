"""Active learning + drift detection for the AMW risk model (pure NumPy).

Two cheap, dependency-light capabilities that keep the learned model honest in a
hostile, shifting domain:

1. **Active learning / human-in-the-loop triage.** The calibrated model emits an
   ``uncertain`` flag (and a ``risk_prob`` near 0.5) for inputs it cannot
   confidently classify. ``select_uncertain`` ranks a batch of predictions by how
   ambiguous they are and ``write_review_queue`` persists the selected items as a
   JSONL review queue a human annotator can label — those gold labels later
   sharpen the decision boundary.

2. **PSI feature drift detection.** The Population Stability Index compares the
   per-feature distribution of a *current* batch against a frozen *baseline*
   (training-time) distribution. A large PSI signals the served traffic has
   shifted away from what the model was trained on (new scam framing, new
   obfuscation) and the model may need retraining. ``feature_stats`` summarizes a
   feature matrix, ``psi`` scores one feature, and ``check_drift`` produces a
   per-feature report flagging drifted features.

Public surface (authoritative ML CONTRACT, DESIGN.md §15):
    select_uncertain(predictions, k) -> list[Prediction]
    write_review_queue(path, items) -> str            # JSONL
    feature_stats(X) -> dict                           # frozen baseline summary
    psi(base, cur) -> float                            # one feature PSI
    check_drift(base, cur, threshold=0.2) -> dict      # per-feature report

Everything here is pure NumPy and NaN-free by construction: degenerate inputs
(empty batches, zero-variance features, empty bins) fall back to well-defined
neutral values rather than raising or producing ``nan``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from app.ml.featurize import INPUT_DIM, numeric_feature_names
from app.ml.types import Prediction

# Number of histogram bins used for the PSI computation. PSI is traditionally
# computed on ~10 deciles; we keep that convention. Bin edges come from the
# baseline so the same partition is applied to both distributions.
_PSI_BINS: int = 10

# Floor added to every bin proportion before the log ratio so an empty bin in
# either distribution can't blow the PSI up to +inf (standard PSI epsilon).
_PSI_EPS: float = 1e-4

# Below this baseline span a feature is treated as (near-)constant: PSI is then
# trivially 0 (no distribution to drift). Avoids degenerate single-bin edges.
_CONST_EPS: float = 1e-9


# --------------------------------------------------------------------------- #
# 1. Active learning — uncertainty-based selection + review queue
# --------------------------------------------------------------------------- #
def _uncertainty_key(p: Prediction) -> tuple:
    """Sort key ranking the *most* ambiguous predictions first.

    Primary: explicitly ``uncertain``-flagged items lead. Secondary: distance of
    the calibrated ``risk_prob`` to the 0.5 decision boundary (closer = more
    ambiguous). The flag comes first so the model's own calibrated uncertain band
    ``[uncertain_low, uncertain_high]`` is always honored even if a flagged item
    happens to sit slightly off 0.5.
    """
    dist = abs(float(p.risk_prob) - 0.5)
    # (-flag) so True (1) sorts before False (0); then ascending distance.
    return (0 if p.uncertain else 1, dist)


def select_uncertain(predictions: Sequence[Prediction], k: int) -> List[Prediction]:
    """Pick the ``k`` most uncertain predictions for human review.

    Ranking (see ``_uncertainty_key``): ``uncertain``-flagged predictions first,
    then by ascending distance of ``risk_prob`` from the 0.5 boundary. Stable —
    ties preserve input order so the selection is deterministic.

    Args:
        predictions: a batch of model ``Prediction`` objects.
        k: maximum number of items to return. ``k <= 0`` yields an empty list;
            ``k`` larger than the batch returns the whole batch (sorted).

    Returns:
        Up to ``k`` predictions, most-ambiguous first.
    """
    if k <= 0 or not predictions:
        return []
    # ``sorted`` is stable, so equal-uncertainty items keep their original order.
    ranked = sorted(predictions, key=_uncertainty_key)
    return ranked[:k]


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses / numpy scalars into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _prediction_to_record(item: Any) -> Dict[str, Any]:
    """Normalize a queue item to a JSON-serializable dict.

    Accepts a ``Prediction`` (flattened to its fields, attributions inlined), any
    other dataclass, a plain mapping, or a fallback ``{"value": repr}`` so the
    writer never raises on a heterogeneous queue.
    """
    if isinstance(item, Prediction):
        record = _jsonable(item)
        # Surface a compact uncertainty score so reviewers can sort the queue.
        record["uncertainty"] = round(0.5 - abs(float(item.risk_prob) - 0.5), 6)
        return record
    if is_dataclass(item) and not isinstance(item, type):
        return _jsonable(item)  # type: ignore[return-value]
    if isinstance(item, Mapping):
        return {str(key): _jsonable(val) for key, val in item.items()}
    return {"value": _jsonable(item)}


def write_review_queue(path: str | Path, items: Sequence[Any]) -> str:
    """Write a human-review queue as JSONL (one record per line).

    Each item is normalized via ``_prediction_to_record`` so a queue of
    ``Prediction`` objects, dicts, or a mix serializes cleanly. UTF-8,
    ``ensure_ascii=False`` so Russian explanations stay readable. The parent
    directory is created if missing.

    Args:
        path: destination ``.jsonl`` file path.
        items: queue items (typically the output of ``select_uncertain``).

    Returns:
        The string path written to.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            record = _prediction_to_record(item)
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return str(out)


# --------------------------------------------------------------------------- #
# 2. PSI feature drift detection
# --------------------------------------------------------------------------- #
def _as_2d(X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce a feature matrix to a 2-D float64 array (N, D)."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def feature_stats(X: np.ndarray | Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Summarize a feature matrix into a frozen baseline for drift comparison.

    Stores, per feature column, the quantile **bin edges** used to partition the
    distribution plus the baseline **bin proportions**, so a later batch can be
    scored against the exact same partition (PSI requires a fixed binning). Also
    keeps simple moments (mean/std/min/max) for reporting.

    Bin edges are the baseline's empirical quantiles (``_PSI_BINS`` bins), which
    makes PSI robust to the very different scales of the hashed block (L2-normed,
    near 0) and the numeric block (raw 0..1 magnitudes). Near-constant columns are
    marked so ``psi`` short-circuits them to 0.

    Args:
        X: a ``(N, D)`` feature matrix (e.g. ``dataset.to_arrays(...)["X"]``).

    Returns:
        A JSON-serializable dict with keys ``n``, ``dim``, and ``features`` (a
        list of per-column dicts: ``edges``, ``base_props``, ``mean``, ``std``,
        ``min``, ``max``, ``constant``).
    """
    arr = _as_2d(X)
    n, dim = (arr.shape if arr.size else (0, 0))

    features: List[Dict[str, Any]] = []
    for j in range(dim):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            features.append(
                {
                    "edges": [],
                    "base_props": [],
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "constant": True,
                }
            )
            continue

        lo = float(col.min())
        hi = float(col.max())
        if (hi - lo) <= _CONST_EPS:
            features.append(
                {
                    "edges": [lo, hi],
                    "base_props": [1.0],
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": lo,
                    "max": hi,
                    "constant": True,
                }
            )
            continue

        # Quantile edges; dedup collapses repeated quantiles (skewed/sparse cols).
        qs = np.linspace(0.0, 1.0, _PSI_BINS + 1)
        edges = np.quantile(col, qs)
        edges = np.unique(edges)
        # Pad the outer edges so an out-of-range current value still falls inside.
        edges[0] = -np.inf
        edges[-1] = np.inf
        base_props = _bin_props(col, edges)
        features.append(
            {
                "edges": [float(e) for e in edges],
                "base_props": [float(p) for p in base_props],
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": lo,
                "max": hi,
                "constant": False,
            }
        )

    return {"n": int(n), "dim": int(dim), "features": features}


def _bin_props(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Proportion of ``values`` falling into each bin defined by ``edges``.

    ``edges`` has ``B + 1`` entries (the outer two are ``-inf`` / ``+inf``);
    returns a length-``B`` vector summing to 1.0 (or all-zero for empty input).
    """
    values = values[np.isfinite(values)]
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n_bins, dtype=np.float64)
    # right=False -> [edge_i, edge_{i+1}); inf edges keep everything in-range.
    idx = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return counts / float(values.size)


def psi(base: Any, cur: np.ndarray | Sequence[float]) -> float:
    """Population Stability Index between a baseline and a current distribution.

    ``PSI = sum_b (cur_b - base_b) * ln(cur_b / base_b)`` over the baseline's bins,
    with an epsilon floor on every proportion so empty bins stay finite. Two call
    styles are supported:

    * ``base`` is a per-feature stats dict (from ``feature_stats``, one column's
      entry) carrying ``edges`` + ``base_props``; ``cur`` is the raw current value
      vector for that feature — it is re-binned with the baseline edges.
    * ``base`` is a 1-D array/list of baseline values; ``cur`` a 1-D array/list of
      current values — both are binned on shared quantile edges derived from
      ``base``.

    Returns 0.0 for a constant / empty baseline (no distribution to drift).

    Rule-of-thumb interpretation: ``< 0.1`` stable, ``0.1–0.25`` moderate shift,
    ``> 0.25`` significant shift.
    """
    if isinstance(base, Mapping):
        edges = np.asarray(base.get("edges", []), dtype=np.float64)
        base_props = np.asarray(base.get("base_props", []), dtype=np.float64)
        if base.get("constant", False) or edges.size < 2 or base_props.size == 0:
            return 0.0
        cur_vals = np.asarray(cur, dtype=np.float64).reshape(-1)
        cur_props = _bin_props(cur_vals, edges)
    else:
        base_vals = np.asarray(base, dtype=np.float64).reshape(-1)
        base_vals = base_vals[np.isfinite(base_vals)]
        cur_vals = np.asarray(cur, dtype=np.float64).reshape(-1)
        cur_vals = cur_vals[np.isfinite(cur_vals)]
        if base_vals.size == 0:
            return 0.0
        lo, hi = float(base_vals.min()), float(base_vals.max())
        if (hi - lo) <= _CONST_EPS:
            return 0.0
        qs = np.linspace(0.0, 1.0, _PSI_BINS + 1)
        edges = np.unique(np.quantile(base_vals, qs))
        edges[0] = -np.inf
        edges[-1] = np.inf
        base_props = _bin_props(base_vals, edges)
        cur_props = _bin_props(cur_vals, edges)

    if base_props.size == 0 or cur_props.size != base_props.size:
        return 0.0

    # Epsilon-floor + renormalize so both remain valid (sum 1) probability vectors.
    b = base_props + _PSI_EPS
    c = cur_props + _PSI_EPS
    b = b / b.sum()
    c = c / c.sum()
    value = float(np.sum((c - b) * np.log(c / b)))
    # PSI is non-negative; clamp away tiny negative float drift.
    return max(0.0, value)


def _feature_label(j: int, dim: int) -> str:
    """Human-readable label for feature column ``j``.

    The trailing ``numeric_dim`` columns map to the engineered numeric feature
    names; the leading ``hash_dim`` columns are anonymous hashed n-gram buckets.
    """
    num_names = numeric_feature_names()
    num_start = dim - len(num_names)
    if num_start >= 0 and j >= num_start:
        return f"числовой:{num_names[j - num_start]}"
    return f"хеш#{j}"


def check_drift(
    base: Mapping[str, Any],
    cur: np.ndarray | Sequence[Sequence[float]],
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """Compare a current feature batch against a frozen baseline and report drift.

    Computes per-feature PSI of ``cur`` against the ``base`` summary produced by
    ``feature_stats``, flags features whose PSI exceeds ``threshold``, and rolls
    up an overall verdict.

    Args:
        base: output of ``feature_stats`` on the training-time feature matrix.
        cur: a ``(N, D)`` current feature matrix (same column order as baseline).
        threshold: per-feature PSI above which a feature is flagged (default 0.2,
            the conventional "significant shift" line).

    Returns:
        A report dict::

            {
              "drifted": bool,                 # any feature over threshold
              "n_drifted": int,
              "max_psi": float,
              "mean_psi": float,
              "threshold": float,
              "n_base": int, "n_cur": int, "dim": int,
              "top": [ {index, name, psi, drifted}, ... ],   # by PSI desc
              "per_feature_psi": [float, ...],               # length D
            }

    Never raises: a dimension mismatch or empty input yields a well-formed report
    with ``drifted=False``.
    """
    feats: List[Mapping[str, Any]] = list(base.get("features", []))
    base_dim = int(base.get("dim", len(feats)))
    arr = _as_2d(cur)
    n_cur, cur_dim = (arr.shape if arr.size else (0, 0))

    dim = min(base_dim, cur_dim) if cur_dim else base_dim
    psis: List[float] = []
    entries: List[Dict[str, Any]] = []
    for j in range(dim):
        stat = feats[j] if j < len(feats) else {}
        col = arr[:, j] if (cur_dim and j < cur_dim) else np.zeros(0)
        value = psi(stat, col) if stat else 0.0
        psis.append(value)
        entries.append(
            {
                "index": j,
                "name": _feature_label(j, base_dim or INPUT_DIM),
                "psi": round(value, 6),
                "drifted": bool(value > threshold),
            }
        )

    if psis:
        max_psi = float(max(psis))
        mean_psi = float(np.mean(psis))
    else:
        max_psi = 0.0
        mean_psi = 0.0
    n_drifted = int(sum(1 for v in psis if v > threshold))
    top = sorted(entries, key=lambda d: d["psi"], reverse=True)[:15]

    return {
        "drifted": bool(n_drifted > 0),
        "n_drifted": n_drifted,
        "max_psi": round(max_psi, 6),
        "mean_psi": round(mean_psi, 6),
        "threshold": float(threshold),
        "n_base": int(base.get("n", 0)),
        "n_cur": int(n_cur),
        "dim": int(dim),
        "top": top,
        "per_feature_psi": [round(v, 6) for v in psis],
    }


__all__ = [
    "select_uncertain",
    "write_review_queue",
    "feature_stats",
    "psi",
    "check_drift",
]
