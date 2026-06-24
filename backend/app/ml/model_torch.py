"""Optional deeper TORCH multimodal fusion model — `model_torch.py` (DESIGN §17).

A true late-fusion multimodal variant of the risk model, behind the SAME
``RiskModel`` Protocol and SAME ``Prediction`` contract as the portable NumPy
student (``model_np.NpRiskModel``). It is a *scale path* for future GPU training;
it is NOT trained by default and the NumPy student remains the always-available
fallback served by ``registry.load_active``.

Architecture (end-to-end trainable):
  * TEXT branch   — the [0:hash_dim) signed-hash n-gram block is treated as a dense
                    bag-of-n-grams; a learned projection + Transformer-style
                    self-attention/MLP block contextualises it -> 256-d.
  * NUMERIC branch— the 32 engineered features -> MLP -> 64-d.
  * VISUAL branch — the per-dna ``visual_scores`` (8 dims, already inside the numeric
                    block) -> projection -> 32-d.
  * FUSION        — concat [256+64+32]=352 -> 2-layer fusion MLP (GELU, LayerNorm,
                    residual) -> 256-d trunk -> identical risk / dims / category
                    heads (sigmoid / sigmoid / softmax), trained with the SAME
                    multi-task weighted, label-distillation loss as the NumPy model.

Hard rule: torch is imported LAZILY inside methods. This module imports cleanly
even when torch is absent; the class only raises a clear ``RuntimeError`` when it
is actually instantiated / used without torch. The wider engine therefore never
breaks for lacking torch — it simply keeps using the NumPy student / rules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.config import clamp_score, risk_level
from app.ml.config import MLConfig, ml_config
from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Prediction,
    RawFeatures,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np  # noqa: F401
    import torch  # noqa: F401


_TORCH_HINT = (
    "model_torch.TorchRiskModel требует установленного PyTorch. "
    "Установите его (`pip install torch`) или используйте портативную "
    "NumPy-модель app.ml.model_np.NpRiskModel (она — основной обслуживаемый артефакт)."
)


def torch_available() -> bool:
    """True, если PyTorch импортируется в текущем окружении (без побочных эффектов)."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment-dependent
        return False


def _require_torch() -> Any:
    """Ленивая загрузка torch; внятная ошибка, если его нет."""
    try:
        import torch  # local, lazy import

        return torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(_TORCH_HINT) from exc


# --------------------------------------------------------------------------- #
# nn.Module factory (built lazily so the module imports without torch).
# --------------------------------------------------------------------------- #
def _build_fusion_module(
    torch: Any,
    *,
    hash_dim: int,
    numeric_dim: int,
    n_dims: int,
    n_cats: int,
    text_dim: int = 256,
    numeric_hidden: int = 64,
    visual_dim: int = 32,
    trunk_dim: int = 256,
    n_heads: int = 4,
    dropout: float = 0.10,
) -> Any:
    """Собрать ``nn.Module`` поздней мультимодальной фьюжн-сети.

    Определяется ВНУТРИ функции, потому что наследование от ``torch.nn.Module``
    невозможно без импортированного torch. Возвращает готовый экземпляр модуля.
    """
    nn = torch.nn
    F = torch.nn.functional

    class _TextBranch(nn.Module):
        """Хэшированный bag-of-n-grams -> проекция -> self-attention -> 256-d."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(hash_dim, text_dim)
            self.norm_in = nn.LayerNorm(text_dim)
            self.attn = nn.MultiheadAttention(
                text_dim, n_heads, dropout=dropout, batch_first=True
            )
            self.norm_attn = nn.LayerNorm(text_dim)
            self.ff = nn.Sequential(
                nn.Linear(text_dim, text_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(text_dim * 2, text_dim),
            )
            self.norm_ff = nn.LayerNorm(text_dim)

        def forward(self, x_text: Any) -> Any:  # (B, hash_dim) -> (B, text_dim)
            h = self.norm_in(self.proj(x_text))
            seq = h.unsqueeze(1)  # (B, 1, text_dim): single token over the n-gram pool
            attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
            h = self.norm_attn(h + attn_out.squeeze(1))
            h = self.norm_ff(h + self.ff(h))
            return h

    class _Fusion(nn.Module):
        """Полная сеть: три ветви -> фьюжн-MLP -> общий ствол -> три головы."""

        def __init__(self) -> None:
            super().__init__()
            self.text = _TextBranch()
            self.numeric = nn.Sequential(
                nn.Linear(numeric_dim, numeric_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(numeric_hidden, numeric_hidden),
                nn.GELU(),
            )
            # Визуальная ветвь читает первые n_dims numeric-слотов (per-dna visual).
            self.visual = nn.Sequential(
                nn.Linear(n_dims, visual_dim),
                nn.GELU(),
            )
            fused_in = text_dim + numeric_hidden + visual_dim
            self.fuse1 = nn.Linear(fused_in, trunk_dim)
            self.fuse_norm1 = nn.LayerNorm(trunk_dim)
            self.fuse2 = nn.Linear(trunk_dim, trunk_dim)
            self.fuse_norm2 = nn.LayerNorm(trunk_dim)
            self.dropout = nn.Dropout(dropout)
            # Многозадачные головы (тот же контракт, что и у NumPy-модели).
            self.head_risk = nn.Linear(trunk_dim, 1)
            self.head_dims = nn.Linear(trunk_dim, n_dims)
            self.head_cat = nn.Linear(trunk_dim, n_cats)

        def forward(self, x: Any) -> Dict[str, Any]:
            """x: (B, hash_dim + numeric_dim). Возвращает ЛОГИТЫ всех голов."""
            x_text = x[:, :hash_dim]
            x_num = x[:, hash_dim:]
            x_vis = x_num[:, :n_dims]  # per-dna visual scores live in slots 0..n_dims

            t = self.text(x_text)
            n = self.numeric(x_num)
            v = self.visual(x_vis)

            fused = torch.cat([t, n, v], dim=-1)
            h = F.gelu(self.fuse_norm1(self.fuse1(fused)))
            h = self.dropout(h)
            h = h + F.gelu(self.fuse_norm2(self.fuse2(h)))  # residual trunk
            h = self.dropout(h)
            return {
                "risk_logit": self.head_risk(h).squeeze(-1),  # (B,)
                "dims_logit": self.head_dims(h),               # (B, n_dims)
                "cat_logit": self.head_cat(h),                 # (B, n_cats)
                "trunk": h,                                     # (B, trunk_dim)
            }

    module = _Fusion()
    return module


# --------------------------------------------------------------------------- #
# Loss (defined lazily; reused by fit).
# --------------------------------------------------------------------------- #
def _multitask_loss(
    torch: Any,
    out: Dict[str, Any],
    *,
    y_risk: Any,
    Y_dims: Any,
    y_cat: Any,
    w: Any,
    cfg: MLConfig,
    focal_gamma: float = 0.0,
) -> Any:
    """Взвешенная многозадачная потеря с дистилляцией мягких целей (DESIGN §4).

    BCE для risk + 8 dims (мягкие цели в [0,1]), кросс-энтропия для категории,
    повзвешенная per-example ``w``. ``focal_gamma>0`` включает фокальную модуляцию
    риск-головы для дисбаланса классов (опционально).
    """
    F = torch.nn.functional
    eps = 1e-7

    p_risk = torch.sigmoid(out["risk_logit"]).clamp(eps, 1.0 - eps)
    bce_risk = -(y_risk * torch.log(p_risk) + (1.0 - y_risk) * torch.log(1.0 - p_risk))
    if focal_gamma > 0.0:
        p_t = torch.where(y_risk >= 0.5, p_risk, 1.0 - p_risk)
        bce_risk = bce_risk * (1.0 - p_t).pow(focal_gamma)
    loss_risk = (w * bce_risk).mean()

    p_dim = torch.sigmoid(out["dims_logit"]).clamp(eps, 1.0 - eps)
    bce_dim = -(Y_dims * torch.log(p_dim) + (1.0 - Y_dims) * torch.log(1.0 - p_dim))
    loss_dim = (w.unsqueeze(-1) * bce_dim).mean()

    logp_cat = F.log_softmax(out["cat_logit"], dim=-1)
    nll_cat = -logp_cat.gather(-1, y_cat.view(-1, 1)).squeeze(-1)
    loss_cat = (w * nll_cat).mean()

    return (
        cfg.risk_loss_weight * loss_risk
        + cfg.dim_loss_weight * loss_dim
        + cfg.cat_loss_weight * loss_cat
    )


# --------------------------------------------------------------------------- #
# The public model.
# --------------------------------------------------------------------------- #
class TorchRiskModel:
    """Опциональная глубокая мультимодальная фьюжн-модель риска (RiskModel Protocol).

    Drop-in замена ``NpRiskModel`` за тем же ``registry.load_active`` контрактом
    и тем же ``Prediction``. torch импортируется ЛЕНИВО: класс можно импортировать
    без torch, но любое создание/использование без torch поднимает ``RuntimeError``.

    Параметры
    ---------
    cfg : MLConfig
        Конфиг жизненного цикла модели (размерности, веса потерь, seed, версия).
    device : str | None
        "cpu" / "cuda"; по умолчанию cuda при наличии, иначе cpu.
    """

    def __init__(self, cfg: MLConfig = ml_config, device: Optional[str] = None) -> None:
        torch = _require_torch()  # raises RuntimeError if torch is absent
        # INPUT_DIM импортируется из featurize (train-side модуль); делаем это
        # лениво, чтобы model_torch импортировался даже до сборки соседних модулей.
        from app.ml.featurize import INPUT_DIM  # noqa: WPS433

        self.cfg = cfg
        self.model_version: str = cfg.version
        self.temperature: float = 1.0  # ставится calibrate; см. predict
        self.input_dim: int = int(INPUT_DIM)
        self.hash_dim: int = int(cfg.hash_dim)
        self.numeric_dim: int = int(cfg.numeric_dim)
        self.category_keys: List[str] = list(CATEGORY_KEYS)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Детерминизм из cfg.seed (DESIGN §18).
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.net = _build_fusion_module(
            torch,
            hash_dim=self.hash_dim,
            numeric_dim=self.numeric_dim,
            n_dims=len(DIMENSION_KEYS),
            n_cats=len(self.category_keys),
            dropout=cfg.dropout,
        ).to(self.device)

    # -- helpers ----------------------------------------------------------- #
    def _to_tensor(self, x: Any) -> Any:
        """np.ndarray | tensor -> float32 tensor на устройстве модели."""
        torch = _require_torch()
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _vectorize(self, features: RawFeatures) -> Any:
        """RawFeatures -> (INPUT_DIM,) numpy через общий featurize.vectorize."""
        from app.ml.featurize import vectorize  # lazy: shared featurizer

        return vectorize(features)

    def _vectorize_batch(self, batch: List[RawFeatures]) -> Any:
        from app.ml.featurize import vectorize_batch  # lazy

        return vectorize_batch(batch)

    # -- training ---------------------------------------------------------- #
    def fit(
        self,
        train: Dict[str, Any],
        val: Optional[Dict[str, Any]],
        cfg: Optional[MLConfig] = None,
        *,
        focal_gamma: float = 0.0,
    ) -> Dict[str, Any]:
        """Сквозное обучение фьюжн-сети (AdamW + cosine LR), best-by-val restore.

        ``train`` / ``val`` — словари из ``dataset.to_arrays`` (ключи X, y_risk,
        Y_dims, y_cat, w). Возвращает history-словарь с по-эпошными потерями.
        Это путь МАСШТАБА (GPU): по умолчанию модель не обучается в пайплайне.
        """
        torch = _require_torch()
        cfg = cfg or self.cfg

        Xtr = self._to_tensor(train["X"])
        ytr = self._to_tensor(train["y_risk"])
        Ytr = self._to_tensor(train["Y_dims"])
        ctr = torch.as_tensor(train["y_cat"], dtype=torch.long, device=self.device)
        wtr = self._to_tensor(train["w"])
        n = Xtr.shape[0]

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        history: Dict[str, Any] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: Optional[Dict[str, Any]] = None

        bs = max(1, cfg.batch_size)
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad(set_to_none=True)
                out = self.net(Xtr[idx])
                loss = _multitask_loss(
                    torch, out,
                    y_risk=ytr[idx], Y_dims=Ytr[idx], y_cat=ctr[idx], w=wtr[idx],
                    cfg=cfg, focal_gamma=focal_gamma,
                )
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach().cpu())
                n_batches += 1
            sched.step()
            train_loss = ep_loss / max(1, n_batches)
            history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if val is not None and len(val.get("X", [])):
                val_loss = self._eval_loss(val, cfg, focal_gamma=focal_gamma)
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            print(
                f"[torch] эпоха {epoch + 1}/{cfg.epochs} "
                f"loss={train_loss:.4f}"
                + (f" val={val_loss:.4f}" if val is not None else "")
            )

        if best_state is not None:  # best-by-val restore
            self.net.load_state_dict(best_state)
        history["best_val_loss"] = best_val if best_state is not None else None
        return history

    def _eval_loss(self, arrays: Dict[str, Any], cfg: MLConfig, *, focal_gamma: float) -> float:
        torch = _require_torch()
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._to_tensor(arrays["X"]))
            loss = _multitask_loss(
                torch, out,
                y_risk=self._to_tensor(arrays["y_risk"]),
                Y_dims=self._to_tensor(arrays["Y_dims"]),
                y_cat=torch.as_tensor(arrays["y_cat"], dtype=torch.long, device=self.device),
                w=self._to_tensor(arrays["w"]),
                cfg=cfg, focal_gamma=focal_gamma,
            )
        return float(loss.detach().cpu())

    # -- inference --------------------------------------------------------- #
    def _forward_probs(self, X: Any) -> Dict[str, Any]:
        """Прямой проход в eval-режиме -> вероятности голов (numpy-ready)."""
        torch = _require_torch()
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._to_tensor(X))
            risk_logit = out["risk_logit"]
            # Температурное масштабирование (если откалибровано).
            t = max(1e-3, float(self.temperature))
            p_risk = torch.sigmoid(risk_logit / t)
            p_dims = torch.sigmoid(out["dims_logit"])
            p_cat = torch.softmax(out["cat_logit"], dim=-1)
        return {
            "p_risk": p_risk.detach().cpu().numpy(),
            "p_dims": p_dims.detach().cpu().numpy(),
            "p_cat": p_cat.detach().cpu().numpy(),
        }

    def _build_prediction(
        self,
        features: RawFeatures,
        p_risk: float,
        p_dims: Any,
        p_cat: Any,
    ) -> Prediction:
        """Собрать ``Prediction`` из вероятностей голов (общий контракт)."""
        cfg = self.cfg
        score = clamp_score(p_risk * 100.0)
        dims = {
            key: clamp_score(float(p_dims[i]) * 100.0)
            for i, key in enumerate(DIMENSION_KEYS)
        }
        cat_idx = int(p_cat.argmax())
        category = self.category_keys[cat_idx]
        confidence = float(p_cat[cat_idx])
        uncertain = cfg.uncertain_low <= p_risk <= cfg.uncertain_high

        attributions = []
        try:  # объяснимость — best-effort, не должна ронять предсказание
            from app.ml.explain import attribute

            attributions = attribute(self, features)
        except Exception:  # pragma: no cover - explain is optional at serve time
            attributions = []

        return Prediction(
            risk_score=score,
            risk_prob=float(p_risk),
            risk_level=risk_level(score),
            dimensions=dims,
            category=category,
            confidence=confidence,
            uncertain=bool(uncertain),
            attributions=attributions,
            model_version=self.model_version,
        )

    def predict(self, features: RawFeatures) -> Prediction:
        """Один ``RawFeatures`` -> калиброванный ``Prediction`` (RiskModel)."""
        import numpy as np  # lazy

        vec = np.asarray(self._vectorize(features), dtype=np.float32).reshape(1, -1)
        probs = self._forward_probs(vec)
        return self._build_prediction(
            features,
            float(probs["p_risk"][0]),
            probs["p_dims"][0],
            probs["p_cat"][0],
        )

    def predict_batch(self, batch: List[RawFeatures]) -> List[Prediction]:
        """Пакетный инференс -> список ``Prediction`` (RiskModel)."""
        if not batch:
            return []
        import numpy as np  # lazy

        X = np.asarray(self._vectorize_batch(batch), dtype=np.float32)
        probs = self._forward_probs(X)
        return [
            self._build_prediction(
                features,
                float(probs["p_risk"][i]),
                probs["p_dims"][i],
                probs["p_cat"][i],
            )
            for i, features in enumerate(batch)
        ]

    # -- serialization ----------------------------------------------------- #
    def save(self, path: str) -> None:
        """Сохранить веса (.pt) + meta json (cfg, T, CATEGORY_KEYS, INPUT_DIM)."""
        import json
        from pathlib import Path

        torch = _require_torch()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict(), "temperature": self.temperature},
            str(p),
        )
        meta = {
            "kind": "torch_fusion",
            "version": self.model_version,
            "temperature": self.temperature,
            "category_keys": self.category_keys,
            "input_dim": self.input_dim,
            "hash_dim": self.hash_dim,
            "numeric_dim": self.numeric_dim,
            "cfg": {
                "hidden": self.cfg.hidden,
                "dropout": self.cfg.dropout,
                "seed": self.cfg.seed,
                "version": self.cfg.version,
            },
        }
        meta_path = p.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str, cfg: MLConfig = ml_config, device: Optional[str] = None) -> "TorchRiskModel":
        """Восстановить ``TorchRiskModel`` из .pt (raises RuntimeError без torch)."""
        torch = _require_torch()
        model = TorchRiskModel(cfg=cfg, device=device)
        blob = torch.load(path, map_location=model.device)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        model.net.load_state_dict(state)
        if isinstance(blob, dict) and "temperature" in blob:
            model.temperature = float(blob["temperature"])
        model.net.eval()
        return model

    # -- scale path: distill back into the portable NumPy student ---------- #
    def distill_to_numpy(self, X: Any) -> Dict[str, Any]:
        """Выдать МЯГКИЕ выходы на ``X`` для дистилляции в NumPy-студента.

        Возвращает словарь с целями ``y_risk`` / ``Y_dims`` / ``y_cat`` (argmax),
        пригодный как мягкая разметка для ``model_np.NpRiskModel.fit`` — чтобы
        сохранить портативный инференс на одном numpy (DESIGN §17).
        """
        import numpy as np  # lazy

        probs = self._forward_probs(np.asarray(X, dtype=np.float32))
        return {
            "y_risk": probs["p_risk"].astype(np.float32),
            "Y_dims": probs["p_dims"].astype(np.float32),
            "y_cat": probs["p_cat"].argmax(axis=-1).astype(np.int64),
            "p_cat": probs["p_cat"].astype(np.float32),
        }
