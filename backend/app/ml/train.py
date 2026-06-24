"""Тренировочный драйвер кастомной риск-модели — ``train(cfg)``.

Полный путь дистилляции (DESIGN §16): построить примеры из синтетики + учителя,
векторизовать в массивы, стратифицированно разбить на train/val, обучить
``NpRiskModel`` (ручной Adam + backprop), откалибровать вероятности по наземной
правде, посчитать валидационные метрики (после калибровки) и сохранить артефакт
через реестр. Возвращает словарь метрик и печатает компактный прогресс по эпохам.

Все RNG детерминированы от ``cfg.seed`` (внутри ``synth`` / ``split`` / ``fit``);
здесь дополнительной случайности нет. ``numpy`` импортируется на верхнем уровне —
это тренировочный модуль, а не путь инференса.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from app.ml import metrics
from app.ml.calibrate import calibrate_model
from app.ml.config import MLConfig, ml_config
from app.ml.dataset import build_examples, split, to_arrays
from app.ml.model_np import NpRiskModel
from app.ml.types import DIMENSION_KEYS


def _scam_rate(y_risk: np.ndarray) -> float:
    """Доля скам-примеров (``y_risk >= 0.5``) в наборе."""
    if y_risk.size == 0:
        return 0.0
    return float(np.mean(y_risk >= 0.5))


def _val_predictions(model: NpRiskModel, val: Dict[str, np.ndarray]):
    """Калиброванные риск-вероятности и предсказания измерений на валидации.

    Возвращает ``(risk_prob (Nv,), dim_probs (Nv,8))``. ``risk_prob`` уже учитывает
    ``model.temperature`` (через ``_forward_probs``), так что метрики калибровки
    (Brier/ECE) считаются ПОСЛЕ температурного масштабирования.
    """
    X = np.asarray(val["X"], dtype=np.float32)
    if X.shape[0] == 0:
        return (np.zeros((0,), dtype=np.float64),
                np.zeros((0, len(DIMENSION_KEYS)), dtype=np.float64))
    risk_prob, dim_probs, _cat = model._forward_probs(X)
    return np.asarray(risk_prob, dtype=np.float64), np.asarray(dim_probs, dtype=np.float64)


def _save_artifact(model: NpRiskModel, metrics_dict: Dict[str, Any],
                   cfg: MLConfig) -> str:
    """Сохранить обученную модель через реестр; деградировать мягко при его отсутствии.

    Контракт: ``registry.save_artifact(model, metrics, cfg) -> path``. Реестр может
    собираться параллельно — если импорт не удался, откатываемся на прямой
    ``model.save`` и возвращаем путь к ``.npz`` (тренировка не должна падать из-за
    несобранного реестра).
    """
    try:
        from app.ml.registry import save_artifact  # ленивый импорт: реестр строится отдельно
    except Exception as exc:  # pragma: no cover - защитный откат
        print(f"[ml] registry недоступен ({exc!r}); fallback на model.save()")
        model.save(cfg.model_path)
        return str(cfg.model_path)

    try:
        path = save_artifact(model, metrics_dict, cfg)
        return str(path)
    except Exception as exc:  # pragma: no cover - защитный откат
        print(f"[ml] save_artifact упал ({exc!r}); fallback на model.save()")
        model.save(cfg.model_path)
        return str(cfg.model_path)


def train(cfg: MLConfig = ml_config) -> Dict[str, Any]:
    """Обучить и сохранить кастомную риск-модель; вернуть словарь метрик.

    Шаги (DESIGN §16): ``build_examples`` -> ``to_arrays`` -> ``split`` ->
    ``NpRiskModel.fit`` -> ``calibrate_model`` -> валидационные метрики ->
    ``registry.save_artifact``. Печатает компактный прогресс. Детерминирован при
    фиксированном ``cfg.seed``.

    Returns:
        Словарь метрик: размеры наборов, история обучения, бинарные метрики риска
        (после калибровки) и ECE, MAE по 8 измерениям ScamDNA, температура
        калибровки, версия модели и путь к артефакту.
    """
    t0 = time.perf_counter()

    # 1) Данные: синтетика (учитель как оракул измерений) + ручные RU-зерна.
    print(f"[ml] сборка примеров (synth_size={cfg.synth_size}, seed={cfg.seed})...")
    examples = build_examples(cfg)
    arrays = to_arrays(examples, cfg)
    n_total = int(arrays["X"].shape[0])
    print(f"[ml] примеров: {n_total}, скам-доля: {_scam_rate(arrays['y_risk']):.3f}, "
          f"INPUT_DIM={arrays['X'].shape[1] if n_total else 0}")

    # 2) Стратифицированное по is_scam разбиение train/val.
    train_arr, val_arr = split(arrays, cfg.val_frac, cfg.seed)
    n_train = int(train_arr["X"].shape[0])
    n_val = int(val_arr["X"].shape[0])
    print(f"[ml] train={n_train} (скам {_scam_rate(train_arr['y_risk']):.3f}) "
          f"val={n_val} (скам {_scam_rate(val_arr['y_risk']):.3f})")

    has_val = n_val > 0

    # 3) Обучение: ручной Adam + backprop, лучшее-по-val восстановление весов.
    model = NpRiskModel(cfg)
    history = model.fit(train_arr, val_arr if has_val else None, cfg)

    # 4) Калибровка по НАЗЕМНОЙ ПРАВДЕ (is_scam синтетики), не по учителю.
    calibrate_model(model, val_arr if has_val else train_arr, cfg)
    temperature = float(getattr(model, "temperature", 1.0))
    print(f"[ml] калибровка ({cfg.calibration}): T={temperature:.4f}")

    # 5) Валидационные метрики ПОСЛЕ калибровки (fallback на train при пустом val).
    eval_arr = val_arr if has_val else train_arr
    y_risk = np.asarray(eval_arr["y_risk"], dtype=np.float64)
    Y_dims = np.asarray(eval_arr["Y_dims"], dtype=np.float64)

    risk_prob, dim_probs = _val_predictions(model, eval_arr)
    bin_metrics = metrics.binary_metrics(y_risk, risk_prob)
    cal_ece = metrics.ece(y_risk, risk_prob)
    per_dim = metrics.dim_metrics(Y_dims, dim_probs)
    dim_mae_mean = (
        float(np.mean([v["mae"] for v in per_dim.values()])) if per_dim else 0.0
    )

    print("[ml] " + metrics.report(
        binary=bin_metrics, e=cal_ece, dims=per_dim,
        title="Валидационные метрики (после калибровки)",
    ))

    metrics_dict: Dict[str, Any] = {
        "version": cfg.version,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "scam_rate": _scam_rate(arrays["y_risk"]),
        "eval_on": "val" if has_val else "train",
        "temperature": temperature,
        "calibration": cfg.calibration,
        "risk": bin_metrics,
        "ece": float(cal_ece),
        "dim_metrics": per_dim,
        "dim_mae_mean": dim_mae_mean,
        "history": {
            "train_loss": list(history.get("train_loss", [])),
            "val_loss": list(history.get("val_loss", [])),
            "val_auroc": list(history.get("val_auroc", [])),
            "val_brier": list(history.get("val_brier", [])),
            "val_dim_mae": list(history.get("val_dim_mae", [])),
            "best_val_loss": history.get("best_val_loss"),
            "best_val_auroc": history.get("best_val_auroc"),
        },
    }

    # 6) Сохранение артефакта через реестр (с мягким откатом).
    artifact_path = _save_artifact(model, metrics_dict, cfg)
    metrics_dict["artifact_path"] = artifact_path

    elapsed = time.perf_counter() - t0
    metrics_dict["train_seconds"] = round(elapsed, 2)
    print(f"[ml] готово за {elapsed:.1f}s — AUROC={bin_metrics['auroc']:.3f} "
          f"Brier={bin_metrics['brier']:.4f} ECE={cal_ece:.4f} "
          f"dim_MAE={dim_mae_mean:.3f} -> {artifact_path}")

    return metrics_dict


__all__: List[str] = ["train"]
