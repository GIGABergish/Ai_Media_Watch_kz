"""Pure-NumPy multi-task scam-risk model — ``NpRiskModel`` (the served student).

A single shared-trunk MLP distilled from the rule engine + synthetic ground
truth (see ``app/ml/DESIGN.md`` §3-§4, §10). Architecture::

    x (INPUT_DIM) -> Linear(hidden) -> ReLU -> Dropout
                  -> risk head:     Linear(1)  + sigmoid
                  -> dims head:     Linear(8)  + sigmoid   (DIMENSION_KEYS order)
                  -> category head: Linear(K)  + softmax   (CATEGORY_KEYS)

Training is manual forward + backprop with Adam, a sample-weighted multi-task
loss (soft-target BCE for risk + dims, cross-entropy for category, L2 on weights
only) and best-by-val weight restore. Inference is dropout-off, applies the
calibration ``temperature`` and reuses ``app.config.risk_level / clamp_score`` so
the model never re-thresholds. The artifact is a small ``.npz`` (weights) plus a
meta JSON; ``load`` rebuilds the model with numpy as the only runtime dependency.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import clamp_score, risk_level
from app.ml import calibrate, explain
from app.ml.config import MLConfig, ml_config
from app.ml.featurize import (
    INPUT_DIM,
    vectorize,
    vectorize_batch,
)
from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Prediction,
    RawFeatures,
)

_EPS = 1e-7  # log-clamp for BCE / cross-entropy


# --------------------------------------------------------------------------- #
# Small numeric helpers (pure numpy, no SciPy)
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable elementwise logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis (stable)."""
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=-1, keepdims=True), _EPS, None)


def _bce(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elementwise binary cross-entropy (soft targets allowed), clamped."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


class _Adam:
    """Minimal Adam optimizer over a dict of named parameter arrays."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class NpRiskModel:
    """Pure-NumPy shared-trunk multi-task risk model (implements RiskModel).

    Parameters are stored as float32 arrays in ``self.params``; ``self.temperature``
    holds the calibration scalar (1.0 = uncalibrated). The number of category
    outputs is ``len(CATEGORY_KEYS)`` and the dims head is fixed at
    ``len(DIMENSION_KEYS) == 8``.
    """

    def __init__(self, cfg: MLConfig = ml_config, *,
                 input_dim: int = INPUT_DIM,
                 init: bool = True):
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.hidden = int(cfg.hidden)
        self.n_dims = len(DIMENSION_KEYS)
        self.n_cat = len(CATEGORY_KEYS)
        self.model_version = cfg.version
        self.temperature: float = 1.0
        self.category_keys: List[str] = list(CATEGORY_KEYS)
        self.params: Dict[str, np.ndarray] = {}
        if init:
            self._init_params(np.random.default_rng(cfg.seed))

    # -- initialization ---------------------------------------------------- #
    def _init_params(self, rng: np.random.Generator) -> None:
        """He-uniform trunk (ReLU, fan_in=input_dim), Xavier heads, zero biases."""
        h, d = self.hidden, self.input_dim

        def he_uniform(fan_in: int, shape) -> np.ndarray:
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, size=shape).astype(np.float32)

        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

        self.params = {
            "W_trunk": he_uniform(d, (d, h)),
            "b_trunk": np.zeros(h, dtype=np.float32),
            "W_risk": xavier(h, 1),
            "b_risk": np.zeros(1, dtype=np.float32),
            "W_dims": xavier(h, self.n_dims),
            "b_dims": np.zeros(self.n_dims, dtype=np.float32),
            "W_cat": xavier(h, self.n_cat),
            "b_cat": np.zeros(self.n_cat, dtype=np.float32),
        }

    # -- forward ----------------------------------------------------------- #
    def _forward(self, X: np.ndarray, *, train: bool,
                 rng: Optional[np.random.Generator] = None) -> dict:
        """Forward pass. Returns a cache of activations for backprop.

        ``X`` is (N, input_dim). In train mode applies inverted dropout with a
        seeded mask; in eval mode dropout is identity (no rescale needed).
        """
        p = self.params
        z1 = X @ p["W_trunk"] + p["b_trunk"]          # (N, hidden)
        a1 = np.maximum(z1, 0.0)                        # ReLU
        relu_mask = (z1 > 0.0)

        drop = self.cfg.dropout
        if train and drop > 0.0:
            keep = 1.0 - drop
            mask = (rng.random(a1.shape) < keep).astype(np.float32) / keep
            h = a1 * mask
        else:
            mask = None
            h = a1

        risk_logit = (h @ p["W_risk"] + p["b_risk"]).reshape(-1)   # (N,)
        dim_logits = h @ p["W_dims"] + p["b_dims"]                  # (N, 8)
        cat_logits = h @ p["W_cat"] + p["b_cat"]                    # (N, K)

        p_risk = _sigmoid(risk_logit)
        p_dims = _sigmoid(dim_logits)
        p_cat = _softmax(cat_logits)

        return {
            "X": X, "h": h, "relu_mask": relu_mask, "drop_mask": mask,
            "risk_logit": risk_logit, "p_risk": p_risk,
            "p_dims": p_dims, "p_cat": p_cat,
        }

    # -- loss + backward --------------------------------------------------- #
    def _loss_and_grads(self, cache: dict, y_risk: np.ndarray,
                        Y_dims: np.ndarray, y_cat: np.ndarray,
                        w: np.ndarray) -> Tuple[float, Dict[str, np.ndarray], dict]:
        """Weighted multi-task loss + full manual backprop grads.

        Returns ``(total_loss, grads, parts)`` where ``parts`` carries the per-task
        loss components. ``y_cat`` is an int index array (N,); ``w`` is the
        per-example sample weight. L2 (weight decay) hits weight matrices only,
        not biases.
        """
        cfg = self.cfg
        p = self.params
        N = cache["X"].shape[0]
        wn = w.reshape(-1).astype(np.float64)
        wsum = max(wn.sum(), _EPS)

        p_risk = cache["p_risk"]
        p_dims = cache["p_dims"]
        p_cat = cache["p_cat"]
        h = cache["h"]

        # ---- losses (weighted means) ----
        l_risk = float((wn * _bce(p_risk, y_risk)).sum() / wsum)
        l_dim = float((wn * _bce(p_dims, Y_dims).mean(axis=1)).sum() / wsum)
        cat_p_true = np.clip(p_cat[np.arange(N), y_cat], _EPS, 1.0)
        l_cat = float((wn * -np.log(cat_p_true)).sum() / wsum)

        l2 = cfg.l2
        reg = 0.0
        if l2 > 0.0:
            for key in ("W_trunk", "W_risk", "W_dims", "W_cat"):
                reg += float(np.sum(p[key].astype(np.float64) ** 2))
            reg *= 0.5 * l2
        loss = (cfg.risk_loss_weight * l_risk
                + cfg.dim_loss_weight * l_dim
                + cfg.cat_loss_weight * l_cat
                + reg)

        # ---- gradients of weighted loss wrt logits ----
        ws = (wn / wsum)[:, None]  # (N,1)

        # BCE+sigmoid -> dL/dlogit = (p - y)
        g_risk_logit = cfg.risk_loss_weight * ws[:, 0] * (p_risk - y_risk)   # (N,)
        g_dim_logits = cfg.dim_loss_weight * ws * (p_dims - Y_dims) / self.n_dims  # (N,8)

        # softmax+CE -> dL/dlogit = p - onehot
        g_cat_logits = p_cat.copy()
        g_cat_logits[np.arange(N), y_cat] -= 1.0
        g_cat_logits = cfg.cat_loss_weight * ws * g_cat_logits               # (N,K)

        grads: Dict[str, np.ndarray] = {}
        grads["W_risk"] = h.T @ g_risk_logit.reshape(-1, 1)
        grads["b_risk"] = np.array([g_risk_logit.sum()], dtype=np.float64)
        grads["W_dims"] = h.T @ g_dim_logits
        grads["b_dims"] = g_dim_logits.sum(axis=0)
        grads["W_cat"] = h.T @ g_cat_logits
        grads["b_cat"] = g_cat_logits.sum(axis=0)

        # backprop into shared hidden h
        dh = (g_risk_logit.reshape(-1, 1) @ p["W_risk"].T
              + g_dim_logits @ p["W_dims"].T
              + g_cat_logits @ p["W_cat"].T)            # (N, hidden)

        # through dropout, then ReLU
        if cache["drop_mask"] is not None:
            dh = dh * cache["drop_mask"]
        dz1 = dh * cache["relu_mask"]                    # (N, hidden)

        grads["W_trunk"] = cache["X"].T @ dz1
        grads["b_trunk"] = dz1.sum(axis=0)

        # L2 on weight matrices
        if l2 > 0.0:
            for key in ("W_trunk", "W_risk", "W_dims", "W_cat"):
                grads[key] = grads[key] + l2 * p[key]

        # cast to float32 to match params
        grads = {k: v.astype(np.float32) for k, v in grads.items()}
        parts = {"loss": loss, "risk": l_risk, "dim": l_dim, "cat": l_cat}
        return loss, grads, parts

    # -- training ---------------------------------------------------------- #
    def fit(self, train: dict, val: Optional[dict], cfg: Optional[MLConfig] = None) -> dict:
        """Train with mini-batch Adam; keep best-by-val weights and restore them.

        ``train`` / ``val`` are the array dicts from ``dataset.to_arrays`` (keys
        ``X, y_risk, Y_dims, y_cat, w``). Returns a per-epoch history dict.
        """
        if cfg is not None:
            self.cfg = cfg
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        X = train["X"].astype(np.float32)
        y_risk = train["y_risk"].astype(np.float64)
        Y_dims = train["Y_dims"].astype(np.float64)
        y_cat = train["y_cat"].astype(np.int64)
        w = train.get("w")
        w = np.ones(X.shape[0], np.float64) if w is None else w.astype(np.float64)

        N = X.shape[0]
        bs = max(1, int(cfg.batch_size))
        opt = _Adam(self.params, cfg.lr)

        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_auroc": [],
            "val_brier": [], "val_dim_mae": [],
        }
        best_score = -np.inf
        best_val_loss = np.inf
        best_params = self._snapshot()

        for epoch in range(int(cfg.epochs)):
            perm = rng.permutation(N)
            ep_loss = 0.0
            ep_seen = 0
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                cache = self._forward(X[idx], train=True, rng=rng)
                loss, grads, _ = self._loss_and_grads(
                    cache, y_risk[idx], Y_dims[idx], y_cat[idx], w[idx])
                opt.step(self.params, grads)
                ep_loss += loss * len(idx)
                ep_seen += len(idx)
            train_loss = ep_loss / max(ep_seen, 1)
            history["train_loss"].append(train_loss)

            # ---- validation ----
            if val is not None and len(val.get("X", [])) > 0:
                vm = self._eval_metrics(val)
                history["val_loss"].append(vm["loss"])
                history["val_auroc"].append(vm["auroc"])
                history["val_brier"].append(vm["brier"])
                history["val_dim_mae"].append(vm["dim_mae"])
                # best-by-val: prefer higher AUROC, tie-break lower loss
                score = vm["auroc"] - 1e-6 * vm["loss"]
                better = (score > best_score) or (
                    np.isclose(score, best_score) and vm["loss"] < best_val_loss)
                if better:
                    best_score = score
                    best_val_loss = vm["loss"]
                    best_params = self._snapshot()
                print(f"[ml] epoch {epoch + 1:02d}/{cfg.epochs} "
                      f"train_loss={train_loss:.4f} val_loss={vm['loss']:.4f} "
                      f"val_auroc={vm['auroc']:.3f} val_brier={vm['brier']:.4f} "
                      f"dim_mae={vm['dim_mae']:.3f}")
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_params = self._snapshot()
                print(f"[ml] epoch {epoch + 1:02d}/{cfg.epochs} "
                      f"train_loss={train_loss:.4f}")

        # restore best-by-val snapshot
        self._restore(best_params)
        history["best_val_loss"] = float(best_val_loss)
        history["best_val_auroc"] = float(best_score if best_score > -np.inf else 0.0)
        return history

    def _eval_metrics(self, arrays: dict) -> dict:
        """Forward (eval mode) over a set and compute loss + risk/dim metrics."""
        X = arrays["X"].astype(np.float32)
        y_risk = arrays["y_risk"].astype(np.float64)
        Y_dims = arrays["Y_dims"].astype(np.float64)
        y_cat = arrays["y_cat"].astype(np.int64)
        w = arrays.get("w")
        w = np.ones(X.shape[0], np.float64) if w is None else w.astype(np.float64)
        cache = self._forward(X, train=False)
        loss, _, _ = self._loss_and_grads(cache, y_risk, Y_dims, y_cat, w)
        p_risk = cache["p_risk"]
        y_bin = (y_risk >= 0.5).astype(np.float64)
        return {
            "loss": float(loss),
            "auroc": _auroc(y_bin, p_risk),
            "brier": float(np.mean((p_risk - y_bin) ** 2)),
            "dim_mae": float(np.mean(np.abs(cache["p_dims"] - Y_dims))),
        }

    def _snapshot(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def _restore(self, snap: Dict[str, np.ndarray]) -> None:
        self.params = {k: v.copy() for k, v in snap.items()}

    # -- inference --------------------------------------------------------- #
    def _forward_probs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eval-mode forward returning (risk_prob, dim_probs, cat_probs).

        The risk probability is calibrated by ``self.temperature`` (logit / T).
        """
        cache = self._forward(X, train=False)
        risk_logit = cache["risk_logit"]
        # temperature scaling on the risk logit -> reuse calibrate.apply on prob
        raw_prob = _sigmoid(risk_logit)
        if self.temperature and self.temperature != 1.0:
            risk_prob = np.array(
                [calibrate.apply(float(pp), float(self.temperature)) for pp in raw_prob])
        else:
            risk_prob = raw_prob
        return risk_prob, cache["p_dims"], cache["p_cat"]

    def predict(self, features: RawFeatures) -> Prediction:
        """Vectorize one ``RawFeatures``, run eval forward, build a ``Prediction``."""
        x = vectorize(features).astype(np.float32).reshape(1, -1)
        risk_prob, p_dims, p_cat = self._forward_probs(x)
        return self._make_prediction(features, float(risk_prob[0]), p_dims[0], p_cat[0])

    def predict_batch(self, batch: List[RawFeatures]) -> List[Prediction]:
        """Vectorize and score a batch of ``RawFeatures`` (one forward pass)."""
        if not batch:
            return []
        X = vectorize_batch(batch).astype(np.float32)
        risk_prob, p_dims, p_cat = self._forward_probs(X)
        return [self._make_prediction(batch[i], float(risk_prob[i]), p_dims[i], p_cat[i])
                for i in range(len(batch))]

    def _make_prediction(self, features: RawFeatures, risk_prob: float,
                         p_dims: np.ndarray, p_cat: np.ndarray) -> Prediction:
        cfg = self.cfg
        score = clamp_score(risk_prob * 100.0)
        dims = {k: clamp_score(float(p_dims[i]) * 100.0)
                for i, k in enumerate(DIMENSION_KEYS)}
        cat_idx = int(np.argmax(p_cat))
        confidence = float(p_cat[cat_idx])
        uncertain = cfg.uncertain_low <= risk_prob <= cfg.uncertain_high
        try:
            attributions = explain.attribute(self, features, top_k=8)
        except Exception:
            attributions = []
        return Prediction(
            risk_score=score,
            risk_prob=float(risk_prob),
            risk_level=risk_level(score),
            dimensions=dims,
            category=self.category_keys[cat_idx] if cat_idx < len(self.category_keys)
            else "",
            confidence=confidence,
            uncertain=bool(uncertain),
            attributions=attributions,
            model_version=self.model_version,
        )

    # -- serialization ----------------------------------------------------- #
    def save(self, path: Optional[str] = None) -> None:
        """Write the ``.npz`` weights and a sibling meta JSON.

        ``path`` defaults to ``cfg.model_path``; the meta JSON is always written
        to ``cfg.meta_path`` (config snapshot, temperature, category keys,
        INPUT_DIM, version).
        """
        cfg = self.cfg
        npz_path = path or cfg.model_path
        np.savez(
            npz_path,
            W_trunk=self.params["W_trunk"], b_trunk=self.params["b_trunk"],
            W_risk=self.params["W_risk"], b_risk=self.params["b_risk"],
            W_dims=self.params["W_dims"], b_dims=self.params["b_dims"],
            W_cat=self.params["W_cat"], b_cat=self.params["b_cat"],
        )
        meta = {
            "version": self.model_version,
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "temperature": float(self.temperature),
            "category_keys": self.category_keys,
            "dimension_keys": list(DIMENSION_KEYS),
            "model_path": str(npz_path),
            "cfg": {
                "hash_dim": cfg.hash_dim,
                "numeric_dim": cfg.numeric_dim,
                "hidden": cfg.hidden,
                "dropout": cfg.dropout,
                "seed": cfg.seed,
                "version": cfg.version,
                "calibration": cfg.calibration,
                "uncertain_low": cfg.uncertain_low,
                "uncertain_high": cfg.uncertain_high,
            },
        }
        with open(cfg.meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Optional[str] = None, cfg: MLConfig = ml_config) -> "NpRiskModel":
        """Reconstruct an ``NpRiskModel`` from a ``.npz`` (+ meta JSON if present)."""
        npz_path = path or cfg.model_path
        meta_path = cfg.meta_path
        meta: dict = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = {}

        input_dim = int(meta.get("input_dim", INPUT_DIM))
        model = NpRiskModel(cfg, input_dim=input_dim, init=False)
        with np.load(npz_path) as data:
            model.params = {
                "W_trunk": data["W_trunk"].astype(np.float32),
                "b_trunk": data["b_trunk"].astype(np.float32),
                "W_risk": data["W_risk"].astype(np.float32),
                "b_risk": data["b_risk"].astype(np.float32),
                "W_dims": data["W_dims"].astype(np.float32),
                "b_dims": data["b_dims"].astype(np.float32),
                "W_cat": data["W_cat"].astype(np.float32),
                "b_cat": data["b_cat"].astype(np.float32),
            }
        model.hidden = model.params["b_trunk"].shape[0]
        model.n_dims = model.params["b_dims"].shape[0]
        model.n_cat = model.params["b_cat"].shape[0]
        model.temperature = float(meta.get("temperature", 1.0))
        model.category_keys = list(meta.get("category_keys", CATEGORY_KEYS))
        model.model_version = str(meta.get("version", cfg.version))
        return model


# --------------------------------------------------------------------------- #
# Local AUROC (kept here so fit's val metric has no cross-module import cycle
# with metrics.py; metrics.py remains the public, richer implementation).
# --------------------------------------------------------------------------- #
def _auroc(y: np.ndarray, prob: np.ndarray) -> float:
    """Mann-Whitney rank AUROC with tie-averaged ranks; 0.5 if one class absent."""
    y = np.asarray(y).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    n_pos = float(np.sum(y >= 0.5))
    n_neg = float(y.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5
    order = np.argsort(prob, kind="mergesort")
    ranks = np.empty(prob.size, dtype=np.float64)
    sorted_p = prob[order]
    i = 0
    n = prob.size
    while i < n:
        j = i
        while j + 1 < n and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based, tie-averaged
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_pos = float(np.sum(ranks[y >= 0.5]))
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
