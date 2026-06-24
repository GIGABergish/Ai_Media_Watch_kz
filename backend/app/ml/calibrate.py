"""Калибровка вероятностей риск-головы — temperature scaling (+ опциональный isotonic).

После обучения нейросеть выдаёт «острые», часто переуверенные вероятности риска.
Калибровка делает `risk_prob` честной вероятностью того, что пример — скам, чтобы
полоса неопределённости ``[uncertain_low, uncertain_high]`` (active learning) и
сравнение с порогами ``app.config`` имели смысл.

Согласно DESIGN §9 калибруемся ПРОТИВ синтетической наземной правды
(``is_scam = y_risk >= 0.5``), а НЕ против меток учителя — так калибратор
исправляет промахи учителя, а не запекает их.

Публичный контракт (всё детерминировано, без обращения к времени):
  * ``fit_temperature(probs, y) -> T``   — поиск температуры золотым сечением по NLL.
  * ``apply(prob, T) -> prob``           — пересчёт вероятности через температуру.
  * ``calibrate_model(model, val, cfg) -> None`` — выставляет ``model.temperature``
    (и при наличии sklearn — опциональный isotonic), НИКОГДА не бросает исключений.

Все вычисления — чистый NumPy; модуль обучающий, поэтому ``import numpy`` на верхнем
уровне допустим (в отличие от inference-пути).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from app.ml.config import MLConfig, ml_config

# Зажим вероятностей перед переводом в логиты, чтобы logit(0)/logit(1) не дал inf.
_EPS = 1e-6
# Границы и точность поиска температуры (NLL унимодальна по T на этом интервале).
_T_LOW = 0.25
_T_HIGH = 10.0
_GOLDEN_TOL = 1e-4
_GOLDEN_MAX_ITER = 100

ArrayLike = Union[Sequence[float], "np.ndarray"]


# --------------------------------------------------------------------------- #
# Базовые преобразования вероятность <-> логит
# --------------------------------------------------------------------------- #
def _to_array(x: ArrayLike) -> "np.ndarray":
    """В 1-D float64-массив (float64 — для устойчивой NLL-арифметики)."""
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _logit(p: "np.ndarray") -> "np.ndarray":
    """log(p / (1 - p)) с зажимом p в [_EPS, 1 - _EPS]."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: "np.ndarray") -> "np.ndarray":
    """Численно устойчивая сигмоида (без переполнения exp на больших |z|)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(logits: "np.ndarray", y: "np.ndarray", T: float) -> float:
    """Бинарный negative log-likelihood меток y при температуре T (среднее по примерам)."""
    p = np.clip(_sigmoid(logits / T), _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


# --------------------------------------------------------------------------- #
# fit_temperature — золотое сечение по NLL
# --------------------------------------------------------------------------- #
def fit_temperature(probs: ArrayLike, y: ArrayLike) -> float:
    """Подобрать температуру T>0, минимизирующую бинарный NLL `y` против `sigmoid(logit(probs)/T)`.

    Реализовано поиском золотым сечением на ``T in [_T_LOW, _T_HIGH]`` (NLL унимодальна
    по T): T>1 «смягчает» переуверенные вероятности, T<1 — «заостряет». Без
    зависимостей и детерминирован.

    Возвращает 1.0 при вырожденном входе (пусто или один класс) — калибровать нечего.
    """
    p = _to_array(probs)
    yy = _to_array(y)
    n = min(p.shape[0], yy.shape[0])
    if n == 0:
        return 1.0
    p, yy = p[:n], yy[:n]
    yy = (yy >= 0.5).astype(np.float64)  # бинаризуем наземную правду на всякий случай
    # Один класс -> NLL минимизируется уходом T в край, что бессмысленно; не калибруем.
    if yy.min() == yy.max():
        return 1.0

    logits = _logit(p)

    # Классический поиск минимума унимодальной функции золотым сечением.
    invphi = (np.sqrt(5.0) - 1.0) / 2.0          # 1/phi  ~ 0.618
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0         # 1/phi^2 ~ 0.382
    a, b = _T_LOW, _T_HIGH
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = _nll(logits, yy, c)
    fd = _nll(logits, yy, d)
    for _ in range(_GOLDEN_MAX_ITER):
        if (b - a) <= _GOLDEN_TOL:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = _nll(logits, yy, c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _nll(logits, yy, d)
    T = 0.5 * (a + b)
    # Безопасность: T должна быть конечной и положительной.
    if not np.isfinite(T) or T <= 0.0:
        return 1.0
    return float(T)


# --------------------------------------------------------------------------- #
# apply — пересчёт вероятности через температуру
# --------------------------------------------------------------------------- #
def apply(prob: Union[float, ArrayLike], T: float) -> Union[float, "np.ndarray"]:
    """Применить температуру: ``sigmoid(logit(prob) / T)``.

    Принимает скаляр или массив; тип возврата соответствует входу (скаляр -> float).
    При невалидной T (<=0 или не конечной) возвращает вход без изменений.
    """
    if not np.isfinite(T) or T <= 0.0:
        T = 1.0
    scalar = np.isscalar(prob)
    p = _to_array(prob)
    out = _sigmoid(_logit(p) / float(T))
    if scalar:
        return float(out[0])
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# Получение НЕкалиброванных вероятностей риска из модели
# --------------------------------------------------------------------------- #
def _uncalibrated_risk_probs(model: object, X: "np.ndarray") -> Optional["np.ndarray"]:
    """Сырые (T=1) вероятности риск-головы модели на матрице признаков X.

    model_np строится параллельно, поэтому пробуем несколько совместимых точек входа
    в порядке предпочтения, не завязываясь на приватные детали:
      1. ``model.risk_probs(X)`` / ``model.predict_proba(X)`` — если модель их отдаёт;
      2. ``model.forward(X, train=False)`` -> dict с ключом риска ('risk'|'p_risk'|...);
      3. ``model.forward_logits(X)`` / ``model._forward(...)`` -> логиты риска.
    Возвращает 1-D массив вероятностей или None, если ничего не подошло.
    """
    Xf = np.asarray(X, dtype=np.float32)

    # 1) Явные вероятностные методы.
    for name in ("risk_probs", "predict_proba", "risk_proba"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                arr = np.asarray(fn(Xf), dtype=np.float64).reshape(-1)
                if arr.size:
                    return np.clip(arr, _EPS, 1.0 - _EPS)
            except Exception:
                pass

    # 2) Полный forward, отдающий словарь голов.
    fwd = getattr(model, "forward", None)
    if callable(fwd):
        out = None
        for kwargs in ({"train": False}, {"training": False}, {}):
            try:
                out = fwd(Xf, **kwargs)
                break
            except TypeError:
                continue
            except Exception:
                out = None
                break
        probs = _risk_from_forward(out)
        if probs is not None:
            return probs

    # 3) Логиты риска напрямую.
    for name in ("forward_logits", "_forward", "risk_logits"):
        fn = getattr(model, name, None)
        if callable(fn):
            for kwargs in ({"train": False}, {"training": False}, {}):
                try:
                    out = fn(Xf, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    out = None
                    break
                logit = _risk_logit_from_forward(out)
                if logit is not None:
                    return np.clip(_sigmoid(logit), _EPS, 1.0 - _EPS)
    return None


def _risk_from_forward(out: object) -> Optional["np.ndarray"]:
    """Достать вероятности риска из результата forward (dict/tuple/array)."""
    val = _extract_head(out, prob_keys=("risk", "p_risk", "risk_prob", "p"))
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    # Если это похоже на логиты (вне [0,1]) — прогоним через сигмоиду.
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = _sigmoid(arr)
    return np.clip(arr, _EPS, 1.0 - _EPS)


def _risk_logit_from_forward(out: object) -> Optional["np.ndarray"]:
    """Достать ЛОГИТЫ риска из результата forward (dict/tuple/array)."""
    val = _extract_head(out, prob_keys=("risk_logit", "z_risk", "logit_risk", "risk", "p_risk"))
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    return arr if arr.size else None


def _extract_head(out: object, prob_keys: Sequence[str]) -> Optional[object]:
    """Универсальный доступ к риск-голове внутри dict/tuple/первого столбца массива."""
    if out is None:
        return None
    if isinstance(out, dict):
        for k in prob_keys:
            if k in out:
                return out[k]
        return None
    if isinstance(out, (tuple, list)):
        return out[0] if out else None
    arr = np.asarray(out)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return arr[:, 0]
    return arr


# --------------------------------------------------------------------------- #
# Опциональный isotonic (sklearn) — мягко пропускаем при отсутствии
# --------------------------------------------------------------------------- #
def _try_isotonic(model: object, probs: "np.ndarray", y: "np.ndarray") -> bool:
    """Обучить изотоническую регрессию вероятностей риска и навесить на модель.

    Требует sklearn; при отсутствии или любой ошибке тихо возвращает False (вызывающий
    откатится на temperature scaling). Калибратор сохраняется в ``model.isotonic`` —
    inference-путь может применить его, если поддерживает.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
    except Exception:
        return False
    try:
        if y.min() == y.max():
            return False
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs, y)
        setattr(model, "isotonic", iso)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# calibrate_model — точка входа, вызываемая train.py
# --------------------------------------------------------------------------- #
def calibrate_model(model: object, val: object, cfg: MLConfig = ml_config) -> None:
    """Откалибровать ``model`` на валидации (in-place), согласно ``cfg.calibration``.

    ``val`` — словарь массивов из ``dataset.split`` с ключами как минимум ``X`` и
    ``y_risk``. Калибруем против НАЗЕМНОЙ ПРАВДЫ ``is_scam = y_risk >= 0.5`` (DESIGN §9):
    исправляем переуверенность, а не запекаем промахи учителя.

    Режимы (``cfg.calibration``):
      * ``"none"``         — ставит ``model.temperature = 1.0``;
      * ``"temperature"``  — temperature scaling золотым сечением;
      * ``"isotonic"``     — изотония через sklearn, иначе откат на temperature.

    НИКОГДА не бросает: при любой проблеме безопасно выставляет ``temperature = 1.0``.
    """
    # Гарантируем валидное состояние даже при раннем выходе.
    try:
        setattr(model, "temperature", 1.0)
    except Exception:
        pass

    mode = (getattr(cfg, "calibration", "temperature") or "temperature").strip().lower()
    if mode in ("none", "off", ""):
        return

    try:
        if not isinstance(val, dict):
            return
        X = val.get("X")
        y_risk = val.get("y_risk")
        if X is None or y_risk is None:
            return
        X = np.asarray(X, dtype=np.float32)
        y = (_to_array(y_risk) >= 0.5).astype(np.float64)
        if X.shape[0] == 0 or y.shape[0] == 0:
            return

        # Сырые (T=1) вероятности риска от модели.
        probs = _uncalibrated_risk_probs(model, X)
        if probs is None or probs.size == 0:
            return
        n = min(probs.shape[0], y.shape[0])
        probs, y = probs[:n], y[:n]
        if y.min() == y.max():
            return  # один класс на валидации — калибровать нечего

        # Изотония (опционально) поверх temperature.
        if mode == "isotonic":
            if _try_isotonic(model, probs, y):
                # Даже при isotonic держим осмысленную T для путей, читающих только её.
                T = fit_temperature(probs, y)
                _set_temperature(model, T)
                return
            # sklearn нет / не вышло -> мягкий откат на temperature scaling.

        T = fit_temperature(probs, y)
        _set_temperature(model, T)
    except Exception:
        # Контракт: никогда не падаем — оставляем нейтральную температуру.
        _set_temperature(model, 1.0)


def _set_temperature(model: object, T: float) -> None:
    """Безопасно выставить ``model.temperature`` валидным положительным числом."""
    try:
        if not np.isfinite(T) or T <= 0.0:
            T = 1.0
        model.temperature = float(T)  # type: ignore[attr-defined]
    except Exception:
        pass
