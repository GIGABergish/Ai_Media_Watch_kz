"""Pure-NumPy evaluation metrics for the multi-task risk model.

Implements correct, dependency-free (numpy only) metrics used by both
``train.py`` (validation reporting) and ``evaluate.py`` (model-vs-teacher
comparison). Everything is NaN-free by construction: degenerate inputs (a class
absent, a zero-variance dimension, an empty bin) fall back to well-defined
neutral values rather than producing ``nan``.

Public surface (authoritative ML CONTRACT):
    binary_metrics(y, prob) -> {auroc, ap, f1, acc, brier}
    ece(y, prob, bins=15) -> float
    dim_metrics(Ytrue, Ypred) -> {dna_key: {mae, corr}}
    report(binary=..., e=..., dims=..., title=...) -> str   # Russian

Design references: DESIGN.md §12.
  * AUROC via the Mann-Whitney rank statistic with average ranks for ties.
  * AP via the precision/recall step integral over descending-prob order.
  * ECE via 15 equal-width [0,1] bins, empty bins contribute 0.
  * Pearson corr clamped to 0.0 for ~zero-variance dimensions.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from app.ml.types import DIMENSION_KEYS

# Numerical floor used to detect (near-)constant vectors / probabilities.
_EPS = 1e-12


def _as_1d_float(a: np.ndarray | list | tuple) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array (stable arithmetic)."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks of ``values`` with ties assigned their average rank.

    Equivalent to ``scipy.stats.rankdata(values, method="average")`` but pure
    NumPy. Used by the Mann-Whitney AUROC so tied probabilities (very common
    with a calibrated model) contribute exactly 0.5 of an ordered pair.
    """
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")  # stable -> deterministic ties
    sorted_vals = values[order]

    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        # Extend the tie group while subsequent values are (numerically) equal.
        while j < n and sorted_vals[j] <= sorted_vals[i] + _EPS:
            j += 1
        # Average of 1-based ranks (i+1 .. j) for the whole tie group.
        avg = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def auroc(y: np.ndarray, prob: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U / rank statistic.

    ``AUROC = (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)`` using average
    ranks for ties. Returns ``0.5`` when either class is absent (single-class
    held-out slice) instead of NaN, as mandated by the contract.
    """
    y = _as_1d_float(y)
    prob = _as_1d_float(prob)
    pos = y >= 0.5
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_ranks(prob)
    sum_pos_ranks = float(ranks[pos].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(y: np.ndarray, prob: np.ndarray) -> float:
    """Average precision = the PR step integral ``sum (R_k - R_{k-1}) * P_k``.

    Thresholds sweep descending probability; precision is taken at each recall
    level reached. Ties in ``prob`` are collapsed onto a single threshold so a
    block of equal scores can't be cherry-ordered to inflate the score. Returns
    ``0.0`` when there are no positives.
    """
    y = _as_1d_float(y)
    prob = _as_1d_float(prob)
    n_pos = float(np.count_nonzero(y >= 0.5))
    if n_pos == 0.0:
        return 0.0

    order = np.argsort(-prob, kind="mergesort")  # descending, stable
    y_sorted = (y[order] >= 0.5).astype(np.float64)
    p_sorted = prob[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1.0 - y_sorted)

    # Collapse tied scores: only keep the last index of each equal-prob block so
    # precision/recall are evaluated once per distinct threshold.
    keep = np.ones(p_sorted.shape[0], dtype=bool)
    keep[:-1] = np.abs(p_sorted[1:] - p_sorted[:-1]) > _EPS

    tp = tp_cum[keep]
    fp = fp_cum[keep]
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, _EPS)

    # Step integral: sum over each recall increment times precision there.
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - prev_recall) * precision))
    return ap


def binary_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """Core binary classification metrics for the risk head.

    Args:
        y: ground-truth labels, 0/1 (or in [0,1], thresholded at 0.5).
        prob: predicted (ideally calibrated) probabilities in [0,1].

    Returns:
        dict with keys ``auroc, ap, f1, acc, brier``. F1/acc are evaluated at a
        fixed 0.5 decision threshold; brier is the mean squared error of the
        probabilities. All values finite (never NaN).
    """
    y = _as_1d_float(y)
    prob = _as_1d_float(prob)
    n = y.shape[0]
    if n == 0:
        return {"auroc": 0.5, "ap": 0.0, "f1": 0.0, "acc": 0.0, "brier": 0.0}

    y_bin = (y >= 0.5).astype(np.float64)
    pred = (prob >= 0.5).astype(np.float64)

    tp = float(np.sum((pred == 1.0) & (y_bin == 1.0)))
    fp = float(np.sum((pred == 1.0) & (y_bin == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y_bin == 1.0)))

    acc = float(np.mean(pred == y_bin))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    brier = float(np.mean((prob - y_bin) ** 2))

    return {
        "auroc": auroc(y_bin, prob),
        "ap": average_precision(y_bin, prob),
        "f1": float(f1),
        "acc": acc,
        "brier": brier,
    }


def ece(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error over ``bins`` equal-width [0,1] bins.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|`` where ``acc_b`` is the empirical
    positive rate and ``conf_b`` the mean predicted probability in bin ``b``.
    Empty bins contribute 0. Probabilities are clipped into [0,1] so out-of-range
    inputs land in the edge bins rather than being dropped.
    """
    y = _as_1d_float(y)
    prob = _as_1d_float(prob)
    n = y.shape[0]
    if n == 0 or bins <= 0:
        return 0.0

    y_bin = (y >= 0.5).astype(np.float64)
    p = np.clip(prob, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed bins so p == 1.0 lands in the last bin (idx bins-1).
    idx = np.searchsorted(edges, p, side="left") - 1
    idx = np.clip(idx, 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = idx == b
        n_b = int(np.count_nonzero(mask))
        if n_b == 0:
            continue  # empty bin contributes 0
        acc_b = float(np.mean(y_bin[mask]))
        conf_b = float(np.mean(p[mask]))
        total += (n_b / n) * abs(acc_b - conf_b)
    return float(total)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 for ~zero-variance inputs (NaN-free)."""
    a = _as_1d_float(a)
    b = _as_1d_float(b)
    if a.shape[0] < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da < _EPS or db < _EPS:
        return 0.0
    r = float(np.sum(a * b) / (da * db))
    # Guard against tiny floating drift outside [-1, 1].
    return float(np.clip(r, -1.0, 1.0))


def dim_metrics(Ytrue: np.ndarray, Ypred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-ScamDNA-dimension regression metrics (MAE + Pearson corr).

    Args:
        Ytrue: (N, 8) ground-truth / teacher soft dimension targets in [0,1].
        Ypred: (N, 8) model dimension predictions in [0,1].

    Returns:
        Ordered dict ``{dna_key: {"mae": ..., "corr": ...}}`` keyed by
        DIMENSION_KEYS. ``corr`` is 0.0 for ~zero-variance columns. Robust to a
        column count smaller/larger than 8 (extra columns ignored; missing ones
        reported as zeros) so it never raises on a malformed batch.
    """
    Yt = np.asarray(Ytrue, dtype=np.float64)
    Yp = np.asarray(Ypred, dtype=np.float64)
    if Yt.ndim == 1:
        Yt = Yt.reshape(-1, 1)
    if Yp.ndim == 1:
        Yp = Yp.reshape(-1, 1)

    out: Dict[str, Dict[str, float]] = {}
    n_cols_t = Yt.shape[1]
    n_cols_p = Yp.shape[1]
    for i, key in enumerate(DIMENSION_KEYS):
        if i >= n_cols_t or i >= n_cols_p or Yt.shape[0] == 0:
            out[key] = {"mae": 0.0, "corr": 0.0}
            continue
        col_t = Yt[:, i]
        col_p = Yp[:, i]
        mae = float(np.mean(np.abs(col_t - col_p)))
        out[key] = {"mae": mae, "corr": _pearson(col_t, col_p)}
    return out


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def report(
    binary: Optional[Mapping[str, float]] = None,
    e: Optional[float] = None,
    dims: Optional[Mapping[str, Mapping[str, float]]] = None,
    title: str = "Метрики модели",
    baseline: Optional[Mapping[str, float]] = None,
) -> str:
    """Render a human-readable Russian metrics report.

    Args:
        binary: output of ``binary_metrics`` for the model.
        e: ECE value (calibration error). Optional.
        dims: output of ``dim_metrics`` (per-dimension MAE/corr). Optional.
        title: section heading.
        baseline: optional second ``binary_metrics`` dict (e.g. the rule teacher)
            rendered alongside the model for comparison, with a delta column.

    Returns:
        Multi-line Russian string suitable for console / MODEL_CARD output.
    """
    lines: list[str] = []
    lines.append(f"=== {title} ===")

    if binary is not None:
        order = ["auroc", "ap", "f1", "acc", "brier"]
        ru = {
            "auroc": "AUROC (площадь под ROC)",
            "ap": "AP (средняя точность)",
            "f1": "F1-мера",
            "acc": "Точность (accuracy)",
            "brier": "Брайер (Brier)",
        }
        if baseline is not None:
            lines.append("Бинарные метрики (риск):")
            lines.append(f"  {'метрика':<26}{'модель':>10}{'учитель':>10}{'дельта':>10}")
            for k in order:
                m = float(binary.get(k, 0.0))
                b = float(baseline.get(k, 0.0))
                lines.append(f"  {ru[k]:<26}{_fmt(m):>10}{_fmt(b):>10}{_fmt(m - b):>10}")
        else:
            lines.append("Бинарные метрики (риск):")
            for k in order:
                lines.append(f"  {ru[k]:<26}{_fmt(float(binary.get(k, 0.0))):>10}")

    if e is not None:
        lines.append(f"Калибровка: ECE = {_fmt(float(e))}")

    if dims:
        lines.append("По измерениям ScamDNA (MAE / корреляция):")
        lines.append(f"  {'измерение':<14}{'MAE':>10}{'corr':>10}")
        # Preserve DIMENSION_KEYS order regardless of dict iteration order.
        keys = [k for k in DIMENSION_KEYS if k in dims] + [
            k for k in dims if k not in DIMENSION_KEYS
        ]
        mae_vals: list[float] = []
        for k in keys:
            d = dims[k]
            mae = float(d.get("mae", 0.0))
            corr = float(d.get("corr", 0.0))
            mae_vals.append(mae)
            lines.append(f"  {k:<14}{_fmt(mae):>10}{_fmt(corr):>10}")
        if mae_vals:
            lines.append(f"  {'среднее MAE':<14}{_fmt(float(np.mean(mae_vals))):>10}")

    return "\n".join(lines)
